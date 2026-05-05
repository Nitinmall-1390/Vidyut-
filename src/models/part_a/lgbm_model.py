"""
VIDYUT Part A — LightGBM Forecasting Model
===========================================================================
LightGBM gradient boosting model for 15-minute demand forecasting.
Uses engineered lag / rolling / calendar / weather features.
Supports quantile regression for prediction intervals.
===========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger
from src.utils.metrics_demand import compute_all_demand_metrics

log = get_logger("vidyut.lgbm_model")


_LGBM_PARAMS_POINT = {
    "objective": "regression_l1",
    "metric": "mape",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "n_estimators": 1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

_LGBM_PARAMS_LOWER = {**_LGBM_PARAMS_POINT, "objective": "quantile", "alpha": 0.10}
_LGBM_PARAMS_UPPER = {**_LGBM_PARAMS_POINT, "objective": "quantile", "alpha": 0.90}


class LGBMForecastModel:
    """
    LightGBM point + interval forecasting model.

    Fits three separate boosters: point estimate (L1), 10th quantile, 90th quantile.
    """

    TARGET_COL = "demand_kw"
    EXCLUDE_COLS = {"timestamp", "demand_kw", "feeder_id", "zone", "archetype"}

    def __init__(
        self,
        feeder_id: str = "all",
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.feeder_id = feeder_id
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model_point: Optional[lgb.LGBMRegressor] = None
        self.model_lower: Optional[lgb.LGBMRegressor] = None
        self.model_upper: Optional[lgb.LGBMRegressor] = None
        self.feature_cols: List[str] = []
        self.is_fitted: bool = False
        self._cat_encoders: Dict[str, LabelEncoder] = {}

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in self.EXCLUDE_COLS]

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._cat_encoders[col] = le
            else:
                if col in self._cat_encoders:
                    le = self._cat_encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x, le=le: (
                            le.transform([x])[0]
                            if x in le.classes_
                            else -1
                        )
                    )
        return df

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "LGBMForecastModel":
        """
        Train point + quantile models.

        Parameters
        ----------
        train_df : pd.DataFrame
            Feature-engineered training DataFrame.
        val_df : pd.DataFrame, optional
            Validation set for early stopping.
        """
        train_df = self._encode_categoricals(train_df, fit=True)
        self.feature_cols = self._get_feature_cols(train_df)

        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df[self.TARGET_COL].values

        fit_kwargs: Dict = {}
        if val_df is not None:
            val_df = self._encode_categoricals(val_df, fit=False)
            X_val = val_df[self.feature_cols].fillna(0)
            y_val = val_df[self.TARGET_COL].values
            callbacks = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=200),
            ]
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "callbacks": callbacks,
            }

        log.info(
            "Fitting LGBMForecastModel | feeder=%s | features=%d | rows=%d",
            self.feeder_id, len(self.feature_cols), len(X_train),
        )

        params_point = {**_LGBM_PARAMS_POINT, "n_estimators": self.n_estimators}
        self.model_point = lgb.LGBMRegressor(**params_point)
        self.model_point.fit(X_train, y_train, **fit_kwargs)

        params_lower = {**_LGBM_PARAMS_LOWER, "n_estimators": self.n_estimators}
        self.model_lower = lgb.LGBMRegressor(**params_lower)
        self.model_lower.fit(X_train, y_train, **fit_kwargs)

        params_upper = {**_LGBM_PARAMS_UPPER, "n_estimators": self.n_estimators}
        self.model_upper = lgb.LGBMRegressor(**params_upper)
        self.model_upper.fit(X_train, y_train, **fit_kwargs)

        self.is_fitted = True
        log.info("LGBMForecastModel fitted successfully.")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        return_intervals: bool = True,
    ) -> pd.DataFrame:
        """
        Generate demand predictions.

        Returns
        -------
        pd.DataFrame with columns: yhat, yhat_lower, yhat_upper
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        df = self._encode_categoricals(df, fit=False)
        X = df[self.feature_cols].fillna(0)

        result = pd.DataFrame(index=df.index)
        result["yhat"] = np.clip(self.model_point.predict(X), 0, None)

        if return_intervals:
            result["yhat_lower"] = np.clip(self.model_lower.predict(X), 0, None)
            result["yhat_upper"] = np.clip(self.model_upper.predict(X), 0, None)
            # Ensure lower ≤ point ≤ upper
            result["yhat_lower"] = result[["yhat_lower", "yhat"]].min(axis=1)
            result["yhat_upper"] = result[["yhat_upper", "yhat"]].max(axis=1)

        return result

    def evaluate(
        self,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Evaluate on test set. Returns full metrics dict."""
        preds = self.predict(test_df)
        y_true = test_df[self.TARGET_COL].values
        return compute_all_demand_metrics(
            y_true,
            preds["yhat"].values,
            preds.get("yhat_lower", pd.Series()).values if "yhat_lower" in preds else None,
            preds.get("yhat_upper", pd.Series()).values if "yhat_upper" in preds else None,
        )

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self.model_point is None:
            return pd.DataFrame()
        imp = self.model_point.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("LGBMForecastModel saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "LGBMForecastModel":
        return joblib.load(path)
