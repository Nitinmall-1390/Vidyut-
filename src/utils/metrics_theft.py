"""
VIDYUT Theft Detection Metrics
===========================================================================
Precision, Recall, F1, AUC-ROC, confusion matrix utilities optimised for
highly imbalanced binary / multi-class theft classification.
===========================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_theft_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Full binary classification metrics for theft vs normal.

    Parameters
    ----------
    y_true : array of {0, 1}
    y_pred : array of {0, 1} (hard predictions)
    y_prob : array of floats (probability of class 1), optional
    threshold : float
        Decision threshold applied to y_prob when y_pred is None.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    fpr = fp / (fp + tn + 1e-10)   # false positive rate ← critical for BESCOM

    metrics: Dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_positive_rate": recall,
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "true_positives": float(tp),
        "false_negatives": float(fn),
        "accuracy": (tp + tn) / len(y_true),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["avg_precision"] = float(
                average_precision_score(y_true, y_prob)
            )
        except ValueError:
            pass

    return metrics


def multiclass_theft_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Multi-class metrics for the XGBoost 3-class classifier
    (normal / theft / technical_loss / billing_error).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names))) if class_names else None

    metrics: Dict[str, object] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass

    return metrics
