"""
VIDYUT Rule Engine
===========================================================================
Hard-coded business-rule flags for electricity theft / anomaly detection.
These rules are non-negotiable transparency requirements for BESCOM — every
alert must be accompanied by at least one human-interpretable rule flag.

Rules:
  R1 — Zero reading for ≥ 5 consecutive days
  R2 — Month-on-month consumption drop > 60%
  R3 — Negative meter reading detected
  R4 — Bill < minimum charge for ≥ 3 consecutive months
  R5 — Consumption spike > mean + 4σ (possible bypass removal)
  R6 — Perfect flat-line reading (meter stuck / tampered)
===========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.feature_config import (
    RULE_MOM_DROP_PCT_THRESHOLD,
    RULE_MIN_CHARGE_MONTHS_THRESHOLD,
    RULE_ZERO_READING_DAYS_THRESHOLD,
)
from src.utils.logger import get_logger

log = get_logger("vidyut.rule_engine")

MINIMUM_MONTHLY_CHARGE_KWH = 30.0   # BESCOM minimum: ~₹100 → ~30 kWh


@dataclass
class RuleFlag:
    """A single rule trigger for a consumer."""
    rule_id: str
    description: str
    triggered: bool
    severity: str   # "HIGH" | "MEDIUM" | "LOW"
    detail: str = ""
    value: Optional[float] = None


@dataclass
class ConsumerRuleResult:
    """All rule evaluation results for a single consumer."""
    consumer_id: str
    flags: List[RuleFlag] = field(default_factory=list)

    @property
    def triggered_flags(self) -> List[RuleFlag]:
        return [f for f in self.flags if f.triggered]

    @property
    def n_triggered(self) -> int:
        return len(self.triggered_flags)

    @property
    def highest_severity(self) -> str:
        sevs = [f.severity for f in self.triggered_flags]
        if "HIGH" in sevs:
            return "HIGH"
        if "MEDIUM" in sevs:
            return "MEDIUM"
        if "LOW" in sevs:
            return "LOW"
        return "NONE"

    def to_dict(self) -> Dict:
        return {
            "consumer_id": self.consumer_id,
            "n_triggered": self.n_triggered,
            "highest_severity": self.highest_severity,
            "flags": [
                {
                    "rule_id": f.rule_id,
                    "description": f.description,
                    "triggered": f.triggered,
                    "severity": f.severity,
                    "detail": f.detail,
                    "value": f.value,
                }
                for f in self.flags
            ],
        }


class RuleEngine:
    """
    Evaluates all business rules for a given consumer's daily consumption series.
    """

    def evaluate_consumer(
        self,
        consumer_id: str,
        daily_consumption: np.ndarray,
    ) -> ConsumerRuleResult:
        """
        Evaluate all rules for one consumer.

        Parameters
        ----------
        consumer_id : str
        daily_consumption : np.ndarray
            Array of daily kWh readings (length = number of days).

        Returns
        -------
        ConsumerRuleResult
        """
        vals = np.asarray(daily_consumption, dtype=float)
        result = ConsumerRuleResult(consumer_id=consumer_id)

        # R1: Zero reading ≥ 5 consecutive days
        max_consec_zeros = _max_consecutive(vals == 0)
        result.flags.append(RuleFlag(
            rule_id="R1",
            description=f"Zero reading ≥ {RULE_ZERO_READING_DAYS_THRESHOLD} consecutive days",
            triggered=max_consec_zeros >= RULE_ZERO_READING_DAYS_THRESHOLD,
            severity="HIGH",
            detail=f"Longest zero streak: {max_consec_zeros} days.",
            value=float(max_consec_zeros),
        ))

        # R2: Month-on-month drop > 60%
        mom_drop = _max_mom_drop_pct(vals)
        result.flags.append(RuleFlag(
            rule_id="R2",
            description=f"Month-on-month consumption drop > {RULE_MOM_DROP_PCT_THRESHOLD}%",
            triggered=mom_drop > RULE_MOM_DROP_PCT_THRESHOLD,
            severity="HIGH",
            detail=f"Max MoM drop: {mom_drop:.1f}%.",
            value=round(mom_drop, 1),
        ))

        # R3: Negative meter reading
        n_negative = int((vals < 0).sum())
        result.flags.append(RuleFlag(
            rule_id="R3",
            description="Negative meter reading detected",
            triggered=n_negative > 0,
            severity="HIGH",
            detail=f"{n_negative} negative readings.",
            value=float(n_negative),
        ))

        # R4: Bill < minimum charge for ≥ 3 months
        low_months = _count_low_month_windows(vals, MINIMUM_MONTHLY_CHARGE_KWH)
        result.flags.append(RuleFlag(
            rule_id="R4",
            description=f"Consumption below minimum charge for ≥ {RULE_MIN_CHARGE_MONTHS_THRESHOLD} months",
            triggered=low_months >= RULE_MIN_CHARGE_MONTHS_THRESHOLD,
            severity="MEDIUM",
            detail=f"{low_months} months below {MINIMUM_MONTHLY_CHARGE_KWH} kWh.",
            value=float(low_months),
        ))

        # R5: Consumption spike > mean + 4σ
        mean_v, std_v = vals[vals > 0].mean() if any(vals > 0) else 0, vals.std()
        spike_threshold = mean_v + 4 * std_v
        n_spikes = int((vals > spike_threshold).sum()) if spike_threshold > 0 else 0
        result.flags.append(RuleFlag(
            rule_id="R5",
            description="Suspicious consumption spike (> mean + 4σ)",
            triggered=n_spikes >= 3,
            severity="MEDIUM",
            detail=f"{n_spikes} spike days (threshold: {spike_threshold:.1f} kWh).",
            value=float(n_spikes),
        ))

        # R6: Perfect flat-line (meter stuck)
        flat_streak = _max_consecutive_equal(vals)
        result.flags.append(RuleFlag(
            rule_id="R6",
            description="Suspiciously flat consumption (meter may be stuck)",
            triggered=flat_streak >= 14,
            severity="MEDIUM",
            detail=f"Longest flat-line streak: {flat_streak} days.",
            value=float(flat_streak),
        ))

        return result

    def apply_rules_batch(
        self,
        stage2_df: pd.DataFrame,
        consumers_raw_df: pd.DataFrame,
        consumer_col: str = "CONS_NO",
        day_prefix: str = "day_",
    ) -> pd.DataFrame:
        """
        Apply rules to all Stage-2 flagged consumers and attach rule columns.

        Parameters
        ----------
        stage2_df : pd.DataFrame
            Output of XGBoostTheftClassifier.predict_with_df().
        consumers_raw_df : pd.DataFrame
            Raw SGCC-format DataFrame with day_* columns.

        Returns
        -------
        stage2_df with added columns: n_rules_triggered, highest_severity, rule_flags_json
        """
        import json

        day_cols = sorted(
            [c for c in consumers_raw_df.columns if c.startswith(day_prefix)],
            key=lambda c: int(c.replace(day_prefix, "")),
        )
        cons_map = {
            str(row[consumer_col]): row[day_cols].values.astype(float)
            for _, row in consumers_raw_df.iterrows()
        }

        rule_results = []
        for _, row in stage2_df.iterrows():
            cid = str(row.get("consumer_id", ""))
            daily_vals = cons_map.get(cid, np.zeros(30))
            res = self.evaluate_consumer(cid, daily_vals)
            rule_results.append({
                "consumer_id": cid,
                "n_rules_triggered": res.n_triggered,
                "highest_severity": res.highest_severity,
                "rule_flags_json": json.dumps([
                    {"rule_id": f.rule_id, "desc": f.description, "detail": f.detail}
                    for f in res.triggered_flags
                ]),
            })

        rules_df = pd.DataFrame(rule_results)
        return stage2_df.merge(rules_df, on="consumer_id", how="left")


# ── Internal helpers ───────────────────────────────────────────────────────

def _max_consecutive(mask: np.ndarray) -> int:
    """Return the length of the longest consecutive True run."""
    max_run = cur = 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _max_consecutive_equal(vals: np.ndarray, round_decimals: int = 2) -> int:
    """Detect flat-line by rounding and checking consecutive equal values."""
    rounded = np.round(vals, round_decimals)
    return _max_consecutive(
        np.diff(rounded, prepend=rounded[0] + 1) == 0
    )


def _max_mom_drop_pct(vals: np.ndarray) -> float:
    """Compute the maximum month-over-month % consumption drop."""
    n = len(vals)
    drops = []
    for start in range(0, n - 30, 30):
        prev_mean = np.mean(vals[start: start + 30])
        curr_mean = np.mean(vals[start + 30: start + 60]) if start + 60 <= n else np.mean(vals[start + 30:])
        if prev_mean > 0 and len(vals[start + 30:]) > 0:
            drops.append(100 * (prev_mean - curr_mean) / prev_mean)
    return float(max(drops)) if drops else 0.0


def _count_low_month_windows(
    vals: np.ndarray, min_kwh: float = 30.0
) -> int:
    """Count how many 30-day windows have total consumption < min_kwh."""
    count = 0
    for start in range(0, len(vals), 30):
        window = vals[start: start + 30]
        if len(window) > 0 and np.sum(window) < min_kwh:
            count += 1
    return count
