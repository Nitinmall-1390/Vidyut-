"""
VIDYUT Confidence Scorer
===========================================================================
Computes a 0-100 confidence score for each theft alert combining:
  - Model prediction probability
  - Number of rule-engine flags triggered
  - SHAP value consistency (optional)

Score bands:
  80–100 → HIGH confidence
  50–79  → MEDIUM confidence
  0–49   → LOW confidence
===========================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.feature_config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from src.utils.logger import get_logger

log = get_logger("vidyut.confidence_scorer")


def compute_confidence_score(
    model_probability: float,
    n_rules_triggered: int = 0,
    max_rules: int = 6,
    shap_consistency: float = 1.0,
    dual_anomaly: bool = True,
) -> int:
    """
    Compute a 0-100 confidence score for a single theft prediction.

    The score is a weighted combination:
      - 50% weight: model predicted probability (normalised to [0, 100])
      - 30% weight: rule-engine signal (n_triggered / max_rules × 100)
      - 10% weight: SHAP consistency (0–1 → 0–100)
      - 10% weight: dual anomaly confirmation (True=100, False=0)

    Parameters
    ----------
    model_probability : float
        Probability of the theft class from XGBoost (0.0–1.0).
    n_rules_triggered : int
        Number of rule-engine flags triggered for this consumer.
    max_rules : int
        Total number of rules in the engine (used for normalisation).
    shap_consistency : float
        A 0–1 value indicating SHAP value alignment with prediction.
        Default 1.0 when not computed.
    dual_anomaly : bool
        Whether both LSTM AE and Isolation Forest flagged this consumer.

    Returns
    -------
    int : score in [0, 100]
    """
    model_score = np.clip(model_probability * 100, 0, 100)
    rules_score = np.clip((n_rules_triggered / max(max_rules, 1)) * 100, 0, 100)
    shap_score = np.clip(shap_consistency * 100, 0, 100)
    dual_score = 100.0 if dual_anomaly else 0.0

    composite = (
        0.50 * model_score
        + 0.30 * rules_score
        + 0.10 * shap_score
        + 0.10 * dual_score
    )
    return int(np.clip(round(composite), 0, 100))


def score_label(score: int) -> str:
    """Return HIGH / MEDIUM / LOW based on score thresholds."""
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    if score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class ConfidenceScorer:
    """Batch confidence scorer for the theft detection pipeline."""

    def score_batch(
        self,
        predictions_df: pd.DataFrame,
        max_rules: int = 6,
    ) -> pd.DataFrame:
        """
        Attach confidence_score and confidence_label columns to a predictions DataFrame.

        Expected columns in predictions_df:
          - prob_theft (float 0-1)
          - n_rules_triggered (int, optional)
          - dual_anomaly (bool, optional)

        Returns
        -------
        pd.DataFrame with added: confidence_score, confidence_label
        """
        df = predictions_df.copy()

        prob_col = "prob_theft" if "prob_theft" in df.columns else None
        rules_col = "n_rules_triggered" if "n_rules_triggered" in df.columns else None
        dual_col = "dual_anomaly" if "dual_anomaly" in df.columns else None

        scores = []
        for _, row in df.iterrows():
            prob = float(row[prob_col]) if prob_col else 0.5
            rules = int(row[rules_col]) if rules_col else 0
            dual = bool(row[dual_col]) if dual_col else True

            score = compute_confidence_score(
                model_probability=prob,
                n_rules_triggered=rules,
                max_rules=max_rules,
                dual_anomaly=dual,
            )
            scores.append(score)

        df["confidence_score"] = scores
        df["confidence_label"] = [score_label(s) for s in scores]
        return df
