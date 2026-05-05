"""
VIDYUT Demand Forecasting Metrics
===========================================================================
MAPE, RMSE, MAE, SMAPE, coverage metrics for demand forecasting evaluation.
===========================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Returns value in [0, 100]."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > epsilon
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error. Returns value in [0, 100]."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R²."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def pinball_loss(
    y_true: np.ndarray,
    y_pred_low: np.ndarray,
    y_pred_high: np.ndarray,
    alpha_low: float = 0.1,
    alpha_high: float = 0.9,
) -> Dict[str, float]:
    """Quantile pinball loss for prediction interval evaluation."""
    y_true = np.asarray(y_true, float)
    loss_low = np.mean(
        np.where(y_true >= y_pred_low,
                 alpha_low * (y_true - y_pred_low),
                 (1 - alpha_low) * (y_pred_low - y_true))
    )
    loss_high = np.mean(
        np.where(y_true >= y_pred_high,
                 alpha_high * (y_true - y_pred_high),
                 (1 - alpha_high) * (y_pred_high - y_true))
    )
    return {"pinball_low": float(loss_low), "pinball_high": float(loss_high)}


def interval_coverage(
    y_true: np.ndarray,
    y_pred_low: np.ndarray,
    y_pred_high: np.ndarray,
) -> float:
    """Fraction of true values within the prediction interval."""
    y_true = np.asarray(y_true, float)
    inside = (y_true >= y_pred_low) & (y_true <= y_pred_high)
    return float(np.mean(inside))


def compute_all_demand_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_low: np.ndarray | None = None,
    y_pred_high: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute the full suite of demand forecasting metrics.

    Returns a flat dict suitable for logging / JSON serialisation.
    """
    metrics: Dict[str, float] = {
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    if y_pred_low is not None and y_pred_high is not None:
        metrics["interval_coverage"] = interval_coverage(
            y_true, y_pred_low, y_pred_high
        )
        metrics.update(
            pinball_loss(y_true, y_pred_low, y_pred_high)
        )
    return metrics
