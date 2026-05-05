"""
VIDYUT SHAP Explainer
===========================================================================
Generates SHAP waterfall plots and feature contribution arrays for both
demand forecasting (LightGBM) and theft detection (XGBoost) models.
===========================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

log = get_logger("vidyut.shap_explainer")


class SHAPExplainer:
    """
    SHAP explanation engine.

    Supports TreeExplainer for LightGBM and XGBoost models.
    Produces per-prediction SHAP values with waterfall plot data.
    """

    def __init__(self, model: object, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self._background_data: Optional[np.ndarray] = None

    def fit(
        self,
        background_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        max_background_samples: int = 500,
    ) -> "SHAPExplainer":
        """
        Initialise the SHAP TreeExplainer with background data.

        Parameters
        ----------
        background_data : pd.DataFrame
            Representative sample of training data for baseline computation.
        feature_cols : list of str, optional
            Feature columns to use. Inferred from background_data if None.
        max_background_samples : int
            Maximum number of background samples (SHAP is O(N²)).
        """
        if feature_cols:
            self.feature_names = feature_cols

        if self.feature_names:
            X_bg = background_data[self.feature_names].fillna(0).values
        else:
            X_bg = background_data.select_dtypes("number").fillna(0).values
            self.feature_names = list(
                background_data.select_dtypes("number").columns
            )

        # Sub-sample background for speed
        if len(X_bg) > max_background_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_bg), max_background_samples, replace=False)
            X_bg = X_bg[idx]

        self._background_data = X_bg

        log.info(
            "Initialising SHAP TreeExplainer | background=%d samples | features=%d",
            len(X_bg), len(self.feature_names),
        )
        self.explainer = shap.TreeExplainer(
            self.model,
            data=X_bg,
            feature_perturbation="interventional",
        )
        return self

    def explain_single(
        self,
        input_df: pd.DataFrame,
        row_index: int = 0,
    ) -> Dict:
        """
        Generate SHAP explanation for a single prediction.

        Returns
        -------
        dict with:
          - shap_values: array of SHAP values (one per feature)
          - base_value: model expected output
          - prediction: model output for this instance
          - top_positive: top 5 features pushing prediction UP
          - top_negative: top 5 features pushing prediction DOWN
          - waterfall_data: list of {feature, value, shap_value} for plot
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit() before explaining.")

        if self.feature_names:
            X = input_df[self.feature_names].fillna(0).values[row_index: row_index + 1]
        else:
            X = input_df.select_dtypes("number").fillna(0).values[row_index: row_index + 1]

        shap_vals = self.explainer.shap_values(X)

        # Handle multi-class output (XGBoost returns list of arrays)
        if isinstance(shap_vals, list):
            # Use the class with highest probability
            try:
                pred_prob = self.model.predict_proba(X)
                top_class = int(np.argmax(pred_prob[0]))
                sv = shap_vals[top_class][0]
                base_val = self.explainer.expected_value[top_class]
                prediction = float(pred_prob[0][top_class])
            except Exception:
                sv = shap_vals[0][0]
                base_val = self.explainer.expected_value[0]
                prediction = float(base_val)
        else:
            sv = shap_vals[0]
            base_val = float(self.explainer.expected_value)
            try:
                prediction = float(self.model.predict(X)[0])
            except Exception:
                prediction = float(base_val + sv.sum())

        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        feature_names = self.feature_names or [f"f{i}" for i in range(len(sv))]
        input_values = X[0]

        waterfall_data = [
            {
                "feature": feature_names[i],
                "input_value": float(input_values[i]),
                "shap_value": float(sv[i]),
                "abs_shap": float(abs(sv[i])),
                "direction": "positive" if sv[i] > 0 else "negative",
            }
            for i in sorted_idx[:20]   # top 20 features
        ]

        top_positive = [d for d in waterfall_data if d["direction"] == "positive"][:5]
        top_negative = [d for d in waterfall_data if d["direction"] == "negative"][:5]

        return {
            "shap_values": sv.tolist(),
            "base_value": float(base_val),
            "prediction": prediction,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "waterfall_data": waterfall_data,
        }

    def explain_batch(
        self,
        input_df: pd.DataFrame,
        max_samples: int = 200,
    ) -> pd.DataFrame:
        """
        Compute mean absolute SHAP values across a batch of predictions.

        Returns
        -------
        pd.DataFrame
            feature | mean_abs_shap | rank
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit() before explaining.")

        if self.feature_names:
            X = input_df[self.feature_names].fillna(0).values[:max_samples]
        else:
            X = input_df.select_dtypes("number").fillna(0).values[:max_samples]

        shap_vals = self.explainer.shap_values(X)

        if isinstance(shap_vals, list):
            # Average absolute SHAP across classes
            abs_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            abs_vals = np.abs(shap_vals)

        mean_abs = abs_vals.mean(axis=0)
        feature_names = self.feature_names or [f"f{i}" for i in range(len(mean_abs))]

        result = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        result["rank"] = result.index + 1
        return result

    def plot_waterfall(
        self,
        explanation: Dict,
        max_display: int = 15,
    ) -> Optional[object]:
        """
        Render a SHAP waterfall chart using matplotlib.

        Returns matplotlib Figure or None if matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            data = explanation["waterfall_data"][:max_display][::-1]  # bottom to top
            features = [d["feature"] for d in data]
            values = [d["shap_value"] for d in data]
            colours = ["#e74c3c" if v > 0 else "#3498db" for v in values]

            fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.5)))
            bars = ax.barh(features, values, color=colours, edgecolor="white", height=0.7)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

            for bar, val in zip(bars, values):
                ax.text(
                    val + (0.002 * max(abs(v) for v in values)),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=9,
                )

            pos_patch = mpatches.Patch(color="#e74c3c", label="Increases prediction")
            neg_patch = mpatches.Patch(color="#3498db", label="Decreases prediction")
            ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

            ax.set_xlabel("SHAP Value (impact on model output)", fontsize=11)
            ax.set_title(
                f"SHAP Waterfall | Prediction: {explanation['prediction']:.4f} "
                f"| Base: {explanation['base_value']:.4f}",
                fontsize=12, pad=15,
            )
            plt.tight_layout()
            return fig
        except Exception as exc:
            log.warning("Could not render waterfall plot: %s", exc)
            return None
