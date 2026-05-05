"""
VIDYUT Part B — Isolation Forest Anomaly Detector
===========================================================================
Stage 1 of the dual unsupervised pipeline.
Uses scikit-learn IsolationForest on consumer aggregate features.
A consumer is flagged anomalous when BOTH IsolationForest AND LSTM AE agree
(intersection strategy → minimises false positives).
===========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.config.settings import get_settings
from src.utils.logger import get_logger

log = get_logger("vidyut.isolation_forest")
settings = get_settings()


class IsolationForestModel:
    """
    Isolation Forest wrapper for consumer-level anomaly detection.

    Operates on aggregate features (mean, std, cv, zero_ratio, etc.)
    produced by build_theft_aggregate_features().
    """

    CONSUMER_COL = "CONS_NO"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[RobustScaler] = None
        self._feature_cols: list = []
        self.is_fitted: bool = False

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        return [c for c in df.columns if c != self.CONSUMER_COL
                and pd.api.types.is_numeric_dtype(df[c])]

    def fit(self, features_df: pd.DataFrame) -> "IsolationForestModel":
        """
        Fit on consumer aggregate features.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of build_theft_aggregate_features(). Includes CONS_NO column.
        """
        self._feature_cols = self._get_feature_cols(features_df)
        X = features_df[self._feature_cols].fillna(0).values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        log.info(
            "Fitting IsolationForest | n=%d consumers | features=%d | "
            "contamination=%.2f",
            len(features_df), len(self._feature_cols), self.contamination,
        )
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        log.info("IsolationForest fitted.")
        return self

    def predict(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score consumers with IsolationForest.

        Returns
        -------
        pd.DataFrame
            consumer_id, if_score (anomaly score, lower=more anomalous),
            if_anomaly (bool)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        X = features_df[self._feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # decision_function: higher = more normal
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)  # 1=normal, -1=anomaly

        result = pd.DataFrame({
            "consumer_id": features_df[self.CONSUMER_COL].values
                if self.CONSUMER_COL in features_df.columns
                else features_df.index.astype(str),
            "if_score": scores,
            "if_anomaly": predictions == -1,
        })
        n_flagged = result["if_anomaly"].sum()
        log.info(
            "IsolationForest: %d / %d consumers flagged (%.1f%%)",
            n_flagged, len(result), 100 * n_flagged / max(len(result), 1),
        )
        return result

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self._feature_cols,
            "config": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_features": self.max_features,
            },
        }, path)
        log.info("IsolationForest saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "IsolationForestModel":
        data = joblib.load(path)
        config = data["config"]
        instance = cls(**config)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance._feature_cols = data["feature_cols"]
        instance.is_fitted = True
        return instance


def dual_anomaly_intersection(
    lstm_scores: pd.DataFrame,
    if_scores: pd.DataFrame,
    consumer_col: str = "consumer_id",
) -> pd.DataFrame:
    """
    Compute the intersection of LSTM AE and Isolation Forest anomaly flags.

    Both models must flag a consumer for them to be passed to Stage 2.
    This intersection strategy minimises false positives.

    Parameters
    ----------
    lstm_scores : pd.DataFrame
        Must contain: consumer_id, lstm_anomaly (bool)
    if_scores : pd.DataFrame
        Must contain: consumer_id, if_anomaly (bool)

    Returns
    -------
    pd.DataFrame
        consumer_id, lstm_anomaly, if_anomaly, dual_anomaly (bool),
        max_recon_error, if_score
    """
    merged = lstm_scores.merge(
        if_scores, on=consumer_col, how="outer"
    ).fillna({"lstm_anomaly": False, "if_anomaly": False})

    merged["dual_anomaly"] = (
        merged["lstm_anomaly"].astype(bool) & merged["if_anomaly"].astype(bool)
    )

    n_lstm = merged["lstm_anomaly"].sum()
    n_if = merged["if_anomaly"].sum()
    n_both = merged["dual_anomaly"].sum()

    log.info(
        "Dual anomaly intersection: LSTM=%d | IF=%d | BOTH=%d (FPR-reduction: %.1f%%)",
        n_lstm, n_if, n_both,
        100 * (1 - n_both / max(max(n_lstm, n_if), 1)),
    )
    return merged
