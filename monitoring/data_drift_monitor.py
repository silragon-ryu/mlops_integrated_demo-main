from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def ks_drift_report(
    baseline_df: pd.DataFrame,
    production_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """
    Column-wise KS test for numeric columns.
    Returns: (per_column_results, drift_detected)
    """
    num_cols = baseline_df.select_dtypes(include=np.number).columns.tolist()
    results: Dict[str, Dict[str, float]] = {}
    drift_detected = False

    for col in num_cols:
        if col not in production_df.columns:
            continue
        a = baseline_df[col].dropna()
        b = production_df[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue

        stat, p = ks_2samp(a, b)
        results[col] = {"ks_stat": float(stat), "p_value": float(p)}
        if p < p_value_threshold:
            drift_detected = True

    return results, drift_detected


def main():
    ap = argparse.ArgumentParser(description="Simple numeric drift monitor (KS test).")
    ap.add_argument("--baseline", default="data/processed/train.csv", help="Baseline CSV path")
    ap.add_argument("--production", required=True, help="Production CSV path")
    ap.add_argument("--p", type=float, default=0.05, help="p-value threshold (default: 0.05)")
    ap.add_argument("--out", default="drift_report.json", help="Output JSON path")
    args = ap.parse_args()

    baseline = load_data(args.baseline)
    prod = load_data(args.production)

    results, drift = ks_drift_report(baseline, prod, p_value_threshold=args.p)

    payload = {"drift_detected": drift, "p_value_threshold": args.p, "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Drift detected: %s", drift)
    logger.info("Report written: %s", args.out)


if __name__ == "__main__":
    main()
