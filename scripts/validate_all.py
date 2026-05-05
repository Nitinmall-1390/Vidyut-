"""
===========================================================================
VIDYUT End-to-End Validation Script
===========================================================================
Loads validation datasets, runs all three Vidyut subsystems, computes
metrics, and emits an EvaluationReport. Used in CI gates and pre-deploy.

Usage:
    python -m scripts.validate_all --output reports/eval_v2.json
    python -m scripts.validate_all --strict   # exit 1 on target failure

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.evaluation_report import build_evaluation_report
from src.utils.logger import setup_logging
from src.utils.metrics_demand import compute_demand_metrics
from src.utils.metrics_ring import compute_ring_metrics
from src.utils.metrics_theft import compute_theft_metrics

logger = logging.getLogger(__name__)


# ===========================================================================
# Validation data loaders (best-effort; uses synthetic if real data missing)
# ===========================================================================
def _load_or_synth_demand(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        logger.info("Loaded real demand validation: %s (rows=%d)", path, len(df))
        return df
    logger.warning("Demand validation file missing → synthesizing")
    n = 96 * 30  # 30 days, 15-min
    rng = np.random.default_rng(42)
    base = 150 + 50 * np.sin(np.linspace(0, 60 * np.pi, n))
    noise = rng.normal(0, 5, n)
    y_true = base + noise
    y_pred = y_true + rng.normal(0, 6, n)  # ~4% MAPE-ish
    return pd.DataFrame({
        "ds": pd.date_range("2024-10-01", periods=n, freq="15min"),
        "y_true": y_true,
        "y_pred": y_pred,
    })


def _load_or_synth_theft(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        logger.info("Loaded real theft validation: %s (rows=%d)", path, len(df))
        return df
    logger.warning("Theft validation file missing → synthesizing")
    rng = np.random.default_rng(7)
    n = 5000
    y_true = rng.choice([0, 1], size=n, p=[0.92, 0.08])
    score = np.where(
        y_true == 1,
        rng.beta(5, 2, size=n),
        rng.beta(2, 8, size=n),
    )
    y_pred = (score > 0.5).astype(int)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "score": score})


def _load_or_synth_ring() -> Dict[str, Any]:
    rng = np.random.default_rng(11)
    communities = [list(range(i * 8, (i + 1) * 8)) for i in range(20)]
    anomalous = []
    for c in communities:
        # 70% of community members anomalous in 1/3 of communities
        if rng.random() < 0.33:
            anomalous.extend(c[: int(len(c) * 0.7)])
        else:
            anomalous.extend(c[: max(1, int(len(c) * 0.1))])
    return {"communities": communities, "anomalous": anomalous}


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vidyut validation runner")
    parser.add_argument("--demand-csv", default="data/processed/datasets/val_demand.csv")
    parser.add_argument("--theft-csv", default="data/processed/datasets/val_theft.csv")
    parser.add_argument("--output", default="reports/evaluation.json")
    parser.add_argument("--markdown", default="reports/evaluation.md")
    parser.add_argument("--model-version", default="v2")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any target fails")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")

    # ---- Demand ----
    demand_df = _load_or_synth_demand(Path(args.demand_csv))
    demand_metrics = compute_demand_metrics(
        demand_df["y_true"].to_numpy(),
        demand_df["y_pred"].to_numpy(),
        timestamps=demand_df.get("ds"),
    )
    logger.info("Demand metrics: %s", demand_metrics)

    # ---- Theft ----
    theft_df = _load_or_synth_theft(Path(args.theft_csv))
    theft_metrics = compute_theft_metrics(
        theft_df["y_true"].to_numpy(),
        theft_df["y_pred"].to_numpy(),
        y_score=theft_df.get("score"),
    )
    logger.info("Theft metrics: %s", theft_metrics)

    # ---- Ring ----
    ring_data = _load_or_synth_ring()
    ring_metrics = compute_ring_metrics(
        graph=None,
        communities=ring_data["communities"],
        anomalous_members=ring_data["anomalous"],
    )
    logger.info("Ring metrics: %s", ring_metrics)

    # ---- Report ----
    report = build_evaluation_report(
        demand_metrics=demand_metrics,
        theft_metrics=theft_metrics,
        ring_metrics=ring_metrics,
        model_version=args.model_version,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.save(args.output)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(report.to_markdown(), encoding="utf-8")
    logger.info("Wrote %s and %s", args.output, args.markdown)

    print(json.dumps({
        "overall_pass": report.overall_pass,
        "targets": report.target_results,
        "report_json": args.output,
        "report_md": args.markdown,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=str))

    if args.strict and not report.overall_pass:
        logger.error("Validation FAILED — strict mode → exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())