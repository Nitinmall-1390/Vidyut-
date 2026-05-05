"""
VIDYUT Batch Predictor
===========================================================================
Orchestrates full inference pipelines for both demand forecasting and
theft detection across multiple feeders / consumers in a single call.
===========================================================================
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    build_demand_features,
    build_theft_aggregate_features,
    build_lstm_sequences,
)
from src.data.ingestion import load_or_generate_weather
from src.explainability.confidence_scorer import ConfidenceScorer
from src.explainability.rule_engine import RuleEngine
from src.models.inference.model_loader import (
    load_demand_ensemble,
    load_isolation_forest,
    load_lstm_autoencoder,
    load_xgboost_classifier,
)
from src.models.part_b.isolation_forest import dual_anomaly_intersection
from src.models.part_b.ring_detector import TheftRingDetector
from src.utils.logger import get_logger

log = get_logger("vidyut.batch_predictor")


class DemandBatchPredictor:
    """
    Runs demand forecasting for a list of feeders.

    Returns a combined DataFrame with feeder_id, timestamp, yhat, etc.
    """

    def __init__(self, feeder_ids: Optional[List[str]] = None) -> None:
        self.feeder_ids = feeder_ids or ["FEEDER_000"]

    def predict(
        self,
        df: pd.DataFrame,
        datetime_col: str = "timestamp",
        value_col: str = "demand_kw",
        feeder_col: str = "feeder_id",
    ) -> pd.DataFrame:
        """
        Generate ensemble demand predictions for all feeders in df.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered demand DataFrame (output of DemandDataLoader).
        """
        all_preds = []

        # Load weather for the date range
        try:
            min_date = pd.to_datetime(df[datetime_col]).min().date()
            max_date = pd.to_datetime(df[datetime_col]).max().date()
            weather_df = load_or_generate_weather(min_date, max_date)
        except Exception:
            weather_df = None

        feeders = df[feeder_col].unique() if feeder_col in df.columns else ["all"]
        for feeder_id in feeders:
            feeder_df = df[df[feeder_col] == feeder_id].copy()
            if len(feeder_df) == 0:
                continue

            # Build features
            feat_df = build_demand_features(
                feeder_df, datetime_col=datetime_col,
                value_col=value_col, weather_df=weather_df,
                feeder_col=feeder_col,
            )
            feat_df = feat_df.dropna(subset=[value_col])

            ensemble = load_demand_ensemble(str(feeder_id))

            if ensemble.is_fitted:
                preds = ensemble.predict(feat_df, datetime_col=datetime_col)
            else:
                log.warning("Ensemble not fitted for feeder=%s. Using mean baseline.", feeder_id)
                preds = pd.DataFrame({
                    "timestamp": feat_df[datetime_col],
                    "yhat": feat_df[value_col].mean(),
                    "yhat_lower": feat_df[value_col].mean() * 0.9,
                    "yhat_upper": feat_df[value_col].mean() * 1.1,
                    "feeder_id": feeder_id,
                })

            all_preds.append(preds)

        if not all_preds:
            return pd.DataFrame()
        return pd.concat(all_preds, ignore_index=True)


class TheftBatchPredictor:
    """
    Runs the full 3-stage theft detection pipeline on a batch of consumers.

    Stage 1: LSTM AE + Isolation Forest (intersection)
    Stage 2: XGBoost classification of flagged consumers
    Stage 3: Ring detection via Louvain community detection
    """

    def __init__(self) -> None:
        self.rule_engine = RuleEngine()
        self.confidence_scorer = ConfidenceScorer()

    def predict(
        self,
        consumers_df: pd.DataFrame,
        consumer_col: str = "CONS_NO",
        lat_col: str = "lat",
        lon_col: str = "lon",
        transformer_col: str = "transformer_id",
    ) -> Dict:
        """
        Run full theft detection pipeline.

        Parameters
        ----------
        consumers_df : pd.DataFrame
            Cleaned SGCC-format DataFrame with day_* consumption columns.

        Returns
        -------
        dict with keys:
          stage1_results, stage2_results, ring_alerts, summary
        """
        log.info("TheftBatchPredictor: running on %d consumers", len(consumers_df))

        # ── Stage 1: Dual Unsupervised ─────────────────────────────────────
        agg_features = build_theft_aggregate_features(consumers_df, consumer_col=consumer_col)

        lstm_model = load_lstm_autoencoder()
        if_model = load_isolation_forest()

        if lstm_model.is_fitted:
            sequences, seq_cids = build_lstm_sequences(consumers_df, consumer_col=consumer_col)
            lstm_scores = lstm_model.score_consumers(sequences, seq_cids)
        else:
            log.warning("LSTM AE not fitted. Marking all as non-anomalous.")
            lstm_scores = pd.DataFrame({
                "consumer_id": agg_features[consumer_col].values,
                "lstm_anomaly": False,
                "max_recon_error": 0.0,
                "mean_recon_error": 0.0,
            })

        if if_model.is_fitted:
            if_results = if_model.predict(agg_features)
        else:
            log.warning("IsolationForest not fitted. Marking all as non-anomalous.")
            if_results = pd.DataFrame({
                "consumer_id": agg_features[consumer_col].values,
                "if_anomaly": False,
                "if_score": 0.0,
            })

        dual_results = dual_anomaly_intersection(lstm_scores, if_results)
        flagged_consumers = dual_results[dual_results["dual_anomaly"]][["consumer_id"]]

        log.info("Stage 1: %d consumers flagged (dual anomaly)", len(flagged_consumers))

        # ── Stage 2: XGBoost Classification ───────────────────────────────
        stage2_results = pd.DataFrame()
        if len(flagged_consumers) > 0:
            xgb_model = load_xgboost_classifier()
            flagged_features = agg_features[
                agg_features[consumer_col].isin(flagged_consumers["consumer_id"])
            ]

            if xgb_model.is_fitted:
                stage2_results = xgb_model.predict_with_df(flagged_features)
            else:
                log.warning("XGBoost not fitted. Assigning all flagged as theft.")
                stage2_results = flagged_features[[consumer_col]].rename(
                    columns={consumer_col: "consumer_id"}
                )
                stage2_results["predicted_class"] = 1
                stage2_results["predicted_label"] = "theft"
                stage2_results["prob_theft"] = 0.7
                stage2_results["confidence_pct"] = 70.0

            # Apply rule engine
            stage2_results = self.rule_engine.apply_rules_batch(
                stage2_results, consumers_df, consumer_col=consumer_col
            )
            # Compute confidence scores
            stage2_results = self.confidence_scorer.score_batch(stage2_results)

        # ── Stage 3: Ring Detection ────────────────────────────────────────
        anomaly_map: Dict[str, bool] = dict(
            zip(dual_results["consumer_id"].astype(str),
                dual_results["dual_anomaly"].astype(bool))
        )

        geo_available = lat_col in consumers_df.columns and lon_col in consumers_df.columns
        tx_available = transformer_col in consumers_df.columns

        ring_detector = TheftRingDetector(
            anomaly_threshold=0.60,
            geohash_precision=6,
        )

        ring_prep = consumers_df[[consumer_col]].rename(
            columns={consumer_col: "consumer_id"}
        )
        if geo_available:
            ring_prep["lat"] = consumers_df[lat_col].values
            ring_prep["lon"] = consumers_df[lon_col].values
        else:
            # Assign synthetic Bangalore coordinates for demo
            import numpy as np
            rng = np.random.default_rng(42)
            ring_prep["lat"] = rng.uniform(12.85, 13.10, len(ring_prep))
            ring_prep["lon"] = rng.uniform(77.45, 77.75, len(ring_prep))

        if tx_available:
            ring_prep["transformer_id"] = consumers_df[transformer_col].values
        else:
            # Assign synthetic transformer IDs
            ring_prep["transformer_id"] = (
                pd.Series(range(len(ring_prep))) // 20
            ).apply(lambda x: f"TX_{x:04d}")

        ring_alerts = ring_detector.run_full_detection(
            ring_prep, anomaly_map,
            consumer_col="consumer_id",
        )

        summary = {
            "total_consumers": len(consumers_df),
            "stage1_flagged": int(dual_results["dual_anomaly"].sum()),
            "stage2_theft": int((stage2_results.get("predicted_class", pd.Series()) == 1).sum()),
            "stage2_technical": int((stage2_results.get("predicted_class", pd.Series()) == 2).sum()),
            "stage2_billing": int((stage2_results.get("predicted_class", pd.Series()) == 3).sum()),
            "rings_detected": len(ring_alerts),
            "high_severity_rings": sum(1 for r in ring_alerts if r.get("severity") == "HIGH"),
        }

        log.info("Theft batch prediction complete: %s", summary)

        return {
            "stage1_results": dual_results,
            "stage2_results": stage2_results,
            "ring_alerts": ring_alerts,
            "ring_detector": ring_detector,
            "summary": summary,
        }
