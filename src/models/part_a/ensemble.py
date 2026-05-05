"""
VIDYUT Part A — Prophet + LightGBM Ensemble
===========================================================================
Combines Prophet (weight 0.4) and LightGBM (weight 0.6) predictions into
a single calibrated demand forecast with uncertainty bounds.
===========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.models.part_a.lgbm_model import LGBMForecastModel
from src.models.part_a.prophet_model import ProphetModel
from src.utils.logger import get_logger
from src.utils.metrics_demand import compute_all_demand_metrics

log = get_logger("vidyut.ensemble")
settings = get_settings()


class DemandEnsemble:
    """
    Weighted ensemble: yhat = 0.4 × Prophet + 0.6 × LightGBM

    Prediction intervals are computed as the weighted average of
    both models' lower/upper bounds.
    """

    def __init__(
        self,
        feeder_id: str,
        prophet_weight: Optional[float] = None,
        lgbm_weight: Optional[float] = None,
    ) -> None:
        self.feeder_id = feeder_id
        self.w_prophet = prophet_weight or settings.prophet_weight
        self.w_lgbm = lgbm_weight or settings.lgbm_weight
        assert abs(self.w_prophet + self.w_lgbm - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.prophet: Optional[ProphetModel] = None
        self.lgbm: Optional[LGBMForecastModel] = None
        self.is_fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        datetime_col: str = "timestamp",
        value_col: str = "demand_kw",
    ) -> "DemandEnsemble":
        """Train both component models."""
        log.info("DemandEnsemble: fitting feeder=%s", self.feeder_id)

        self.prophet = ProphetModel(feeder_id=self.feeder_id)
        self.prophet.fit(train_df, datetime_col=datetime_col, value_col=value_col)

        self.lgbm = LGBMForecastModel(feeder_id=self.feeder_id)
        self.lgbm.fit(train_df, val_df=val_df)

        self.is_fitted = True
        log.info("DemandEnsemble fitted for feeder=%s", self.feeder_id)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        datetime_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, yhat, yhat_lower, yhat_upper,
                     prophet_yhat, lgbm_yhat
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted.")

        # Prophet predictions
        prophet_preds = self.prophet.predict(df, datetime_col=datetime_col)
        prophet_preds = prophet_preds.rename(
            columns={"ds": "timestamp", "yhat": "prophet_yhat",
                     "yhat_lower": "prophet_lower", "yhat_upper": "prophet_upper"}
        )

        # LightGBM predictions
        lgbm_preds = self.lgbm.predict(df)
        lgbm_preds = lgbm_preds.rename(
            columns={"yhat": "lgbm_yhat",
                     "yhat_lower": "lgbm_lower", "yhat_upper": "lgbm_upper"}
        )

        # Align lengths
        min_len = min(len(prophet_preds), len(lgbm_preds), len(df))
        prophet_preds = prophet_preds.iloc[:min_len]
        lgbm_preds = lgbm_preds.iloc[:min_len]

        # Weighted combination
        result = pd.DataFrame()
        result["timestamp"] = df[datetime_col].values[:min_len]
        result["prophet_yhat"] = prophet_preds["prophet_yhat"].values
        result["lgbm_yhat"] = lgbm_preds["lgbm_yhat"].values
        result["yhat"] = np.clip(
            (self.w_prophet * result["prophet_yhat"] + self.w_lgbm * result["lgbm_yhat"]),
            0, None
        )

        if "prophet_lower" in prophet_preds.columns and "lgbm_lower" in lgbm_preds.columns:
            result["yhat_lower"] = np.clip(
                (self.w_prophet * prophet_preds["prophet_lower"].values + self.w_lgbm * lgbm_preds["lgbm_lower"].values),
                0, None
            )
            result["yhat_upper"] = np.clip(
                (self.w_prophet * prophet_preds["prophet_upper"].values + self.w_lgbm * lgbm_preds["lgbm_upper"].values),
                0, None
            )

        result["feeder_id"] = self.feeder_id
        return result

    def evaluate(
        self,
        test_df: pd.DataFrame,
        datetime_col: str = "timestamp",
        value_col: str = "demand_kw",
    ) -> Dict[str, float]:
        """Evaluate ensemble on a held-out test set."""
        preds = self.predict(test_df, datetime_col=datetime_col)
        y_true = test_df[value_col].values[: len(preds)]
        y_pred = preds["yhat"].values
        y_low = preds.get("yhat_lower", pd.Series(np.zeros(len(preds)))).values
        y_high = preds.get("yhat_upper", pd.Series(np.zeros(len(preds)))).values
        metrics = compute_all_demand_metrics(y_true, y_pred, y_low, y_high)
        log.info("Ensemble metrics for feeder=%s: %s", self.feeder_id, metrics)
        return metrics

    def save(self, base_dir: Path) -> None:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.prophet.save(base_dir / f"{self.feeder_id}_prophet.joblib")
        self.lgbm.save(base_dir / f"{self.feeder_id}_lgbm.joblib")
        meta = {
            "feeder_id": self.feeder_id,
            "w_prophet": self.w_prophet,
            "w_lgbm": self.w_lgbm,
        }
        joblib.dump(meta, base_dir / f"{self.feeder_id}_ensemble_meta.joblib")
        log.info("DemandEnsemble saved to %s", base_dir)

    @classmethod
    def load(cls, base_dir: Path, feeder_id: str) -> "DemandEnsemble":
        base_dir = Path(base_dir)
        meta = joblib.load(base_dir / f"{feeder_id}_ensemble_meta.joblib")
        instance = cls(
            feeder_id=meta["feeder_id"],
            prophet_weight=meta["w_prophet"],
            lgbm_weight=meta["w_lgbm"],
        )
        instance.prophet = ProphetModel.load(base_dir / f"{feeder_id}_prophet.joblib")
        instance.lgbm = LGBMForecastModel.load(base_dir / f"{feeder_id}_lgbm.joblib")
        instance.is_fitted = True
        return instance
