"""
VIDYUT Part B — XGBoost 3-Class Theft Classifier
===========================================================================
Stage 2: Classifies dual-flagged consumers into:
  0 = normal  |  1 = theft  |  2 = technical_loss  |  3 = billing_error

Uses SMOTE oversampling to handle extreme class imbalance.
===========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.config.feature_config import THEFT_CLASS_LABELS
from src.utils.logger import get_logger
from src.utils.metrics_theft import multiclass_theft_metrics

log = get_logger("vidyut.xgboost_classifier")

_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 4,
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


class XGBoostTheftClassifier:
    """
    XGBoost multi-class classifier for theft type attribution.

    Pipeline: RobustScaler → SMOTE → XGBClassifier
    """

    CONSUMER_COL = "CONS_NO"
    EXCLUDE_COLS = {"CONS_NO", "consumer_id"}

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 30,
        smote_k_neighbors: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.smote_k_neighbors = smote_k_neighbors

        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[RobustScaler] = None
        self._feature_cols: List[str] = []
        self.is_fitted: bool = False

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns
                if c not in self.EXCLUDE_COLS
                and pd.api.types.is_numeric_dtype(df[c])]

    def fit(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        val_features: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.Series] = None,
    ) -> "XGBoostTheftClassifier":
        """
        Train the classifier with SMOTE augmentation.

        Parameters
        ----------
        features_df : pd.DataFrame
            Aggregate theft features (output of build_theft_aggregate_features).
        labels : pd.Series
            0=normal, 1=theft, 2=technical_loss, 3=billing_error
        """
        self._feature_cols = self._get_feature_cols(features_df)
        X = features_df[self._feature_cols].fillna(0).values
        y = labels.values.astype(int)

        # Scale
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # SMOTE (minority class oversampling)
        log.info("Applying SMOTE for class balancing…")
        class_counts = pd.Series(y).value_counts()
        log.info("Pre-SMOTE class distribution: %s", class_counts.to_dict())

        # Only apply SMOTE if there are enough minority samples
        min_class_count = class_counts.min()
        k = min(self.smote_k_neighbors, min_class_count - 1)
        if k >= 1:
            smote = SMOTE(k_neighbors=k, random_state=42)
            try:
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
                log.info(
                    "Post-SMOTE: %d → %d samples",
                    len(X_scaled), len(X_resampled),
                )
            except Exception as e:
                log.warning("SMOTE failed (%s). Using original data.", e)
                X_resampled, y_resampled = X_scaled, y
        else:
            log.warning("Insufficient samples for SMOTE. Using original data.")
            X_resampled, y_resampled = X_scaled, y

        # Fit model
        params = {
            **_XGB_PARAMS,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
        self.model = xgb.XGBClassifier(**params)

        fit_kwargs: Dict = {}
        if val_features is not None and val_labels is not None:
            X_val = self.scaler.transform(
                val_features[self._feature_cols].fillna(0).values
            )
            fit_kwargs = {
                "eval_set": [(X_val, val_labels.values.astype(int))],
                "early_stopping_rounds": self.early_stopping_rounds,
                "verbose": False,
            }

        log.info(
            "Training XGBoost classifier | samples=%d | features=%d | classes=4",
            len(X_resampled), len(self._feature_cols),
        )
        self.model.fit(X_resampled, y_resampled, **fit_kwargs)
        self.is_fitted = True
        log.info("XGBoost classifier trained.")
        return self

    def predict(
        self, features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and probabilities.

        Returns
        -------
        y_pred : np.ndarray of int   — class label {0,1,2,3}
        y_prob : np.ndarray (N, 4)   — class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = self.scaler.transform(
            features_df[self._feature_cols].fillna(0).values
        )
        y_prob = self.model.predict_proba(X)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred, y_prob

    def predict_with_df(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict and return a DataFrame with consumer_id + class info.
        """
        y_pred, y_prob = self.predict(features_df)
        result = pd.DataFrame()
        if "CONS_NO" in features_df.columns:
            result["consumer_id"] = features_df["CONS_NO"].values
        result["predicted_class"] = y_pred
        result["predicted_label"] = [THEFT_CLASS_LABELS.get(int(c), "unknown") for c in y_pred]
        result["prob_normal"] = y_prob[:, 0]
        result["prob_theft"] = y_prob[:, 1]
        result["prob_technical"] = y_prob[:, 2]
        result["prob_billing"] = y_prob[:, 3]
        result["confidence_pct"] = (y_prob.max(axis=1) * 100).round(1)
        return result

    def evaluate(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict:
        """Return full multi-class metrics."""
        y_pred, y_prob = self.predict(features_df)
        class_names = [THEFT_CLASS_LABELS[i] for i in sorted(THEFT_CLASS_LABELS)]
        return multiclass_theft_metrics(
            labels.values, y_pred, class_names=class_names, y_prob=y_prob
        )

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self._feature_cols,
        }, path)
        log.info("XGBoostTheftClassifier saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostTheftClassifier":
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance._feature_cols = data["feature_cols"]
        instance.is_fitted = True
        return instance
