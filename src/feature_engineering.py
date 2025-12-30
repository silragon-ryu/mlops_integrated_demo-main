from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from src.utils import add_date_features, add_hashed_features

DEFAULT_BUCKETS = {
    "item": 2000,
    "branch": 200,
    "invoice": 5000,
    "item_branch": 5000,
    "item_month": 5000,
}


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize raw sales data."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Date", "BranchID", "InvoiceNumber", "ItemCode", "QuantitySold"]
    if any(c not in df.columns for c in required):
        raise ValueError("Missing required columns")

    start = len(df)
    out = df[required].copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    for c in ["BranchID", "InvoiceNumber", "ItemCode"]:
        out[c] = out[c].astype(str).str.strip()

    out["QuantitySold"] = pd.to_numeric(out["QuantitySold"], errors="coerce")
    out = out.dropna(subset=["QuantitySold"])
    out = out[out["QuantitySold"] > 0]

    out = out.drop_duplicates(
        subset=["Date", "BranchID", "InvoiceNumber", "ItemCode", "QuantitySold"]
    )

    out = out.sort_values("Date").reset_index(drop=True)
    print(f"Cleaned: {start} → {len(out)} rows")

    return out


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological train/test split."""
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")

    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)

    if n < 2:
        raise ValueError("Need at least 2 rows to split")

    split_idx = int(n * (1 - test_frac))
    split_idx = min(max(split_idx, 1), n - 1)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    cutoff = df.loc[split_idx - 1, "Date"]

    return train, test, cutoff


def compute_thresholds(y: np.ndarray, q_low=0.33, q_high=0.66):
    """Compute train-only quantile thresholds."""
    low = float(np.quantile(y, q_low))
    high = float(np.quantile(y, q_high))

    if low == high:
        low = float(np.quantile(y, 0.5))
        high = float(np.quantile(y, 0.9))
        if low == high:
            high = low + 1e-9

    return low, high


def bucketize(df: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    out = df.copy()
    y = out["QuantitySold"].astype(float).to_numpy()

    y_class = np.where(y <= low, 0, np.where(y <= high, 1, 2)).astype(int)

    out["y_class"] = y_class
    out["quantity_class"] = pd.Series(y_class).map({
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }).values

    return out



def class_weights(y: np.ndarray, n_classes: int = 3):
    """Balanced class weights (always returns all classes)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    total = counts.sum()

    weights = {i: (total / (n_classes * c)) if c > 0 else 0.0 for i, c in enumerate(counts)}

    positive = [w for w in weights.values() if w > 0]
    mean = float(np.mean(positive)) if positive else 1.0

    weights = {k: (v / mean if v > 0 else 0.0) for k, v in weights.items()}
    return weights, counts



def _run_pipeline(in_path: str, out_clean: str, out_train: str, out_test: str, out_art: str) -> None:
        in_path = os.getenv("SALES_PATH", "data/raw/sales_raw.csv")

        out_clean = "data/raw/clean_sales.csv"
        out_train = "data/processed/train.csv"
        out_test = "data/processed/test.csv"
        out_art = "data/processed/feature_artifacts.json"

        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        print("Feature Engineering Pipeline: ")

        # Load
        if in_path.endswith(".csv"):
            df_raw = pd.read_csv(in_path, sep=";", dtype=str, engine="python")
        else:
            df_raw = pd.read_excel(in_path, dtype=str)

        raw_rows = len(df_raw)
        raw_cols = len(df_raw.columns)
        print(f"Loaded {raw_rows} rows")

        # Clean
        df = clean_sales(df_raw)
        df.to_csv(out_clean, index=False)

        # Split
        train, test, cutoff = time_split(df)
        print(f"Split: {len(train)} train / {len(test)} test")

        # Bucketing
        low, high = compute_thresholds(train["QuantitySold"].to_numpy())
        train = bucketize(train, low, high)
        test = bucketize(test, low, high)

        # Features
        train = add_date_features(train)
        test = add_date_features(test)

        train = add_hashed_features(train, DEFAULT_BUCKETS)
        test = add_hashed_features(test, DEFAULT_BUCKETS)


        keep_cols = [
            "y_class",
            "QuantitySold",
            "year",
            "month",
            "day",
            "dayofweek",
            "is_weekend",
            "h_item",
            "h_branch",
            "h_invoice",
            "h_item_branch",
            "h_item_month",
        ]

        train[keep_cols].to_csv(out_train, index=False)
        test[keep_cols].to_csv(out_test, index=False)

        # Artifacts
        weights, counts = class_weights(train["y_class"].to_numpy())

        artifacts = {
            "input": {
                "path": in_path,
                "total_rows_loaded": raw_rows,
                "total_columns_loaded": raw_cols,
            },
            "split": {
                "cutoff_date": str(cutoff),
                "train_rows": len(train),
                "test_rows": len(test),
            },
            "thresholds": {"low": low, "high": high},
            "class_counts": {str(i): int(c) for i, c in enumerate(counts)},
            "class_weights": {str(k): float(v) for k, v in weights.items()},
            "hash_buckets": DEFAULT_BUCKETS,
            "features": {
                "numeric_features": [
                    "QuantitySold", "year", "month", "day", "dayofweek", "is_weekend",
                    "h_item", "h_branch", "h_invoice", "h_item_branch", "h_item_month"
                ],
                "target": "y_class",
            }
        }

        with open(out_art, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)

        print("Pipeline completed")

def run_feature_engineering(in_path: str | None = None, out_dir: str | None = None) -> dict:
    """Programmatic entrypoint used by the Prefect pipeline."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if in_path is None:
        in_path = os.path.join(project_root, "data", "raw", "sales_raw.csv")
        maybe_clean = os.path.join(project_root, "data", "raw", "clean_sales.csv")
        if os.path.exists(maybe_clean):
            in_path = maybe_clean
    if out_dir is None:
        out_dir = os.path.join(project_root, "data", "processed")

    os.makedirs(out_dir, exist_ok=True)
    out_clean = os.path.join(project_root, "data", "raw", "clean_sales.csv")
    out_train = os.path.join(out_dir, "train.csv")
    out_test = os.path.join(out_dir, "test.csv")
    out_art = os.path.join(out_dir, "feature_artifacts.json")

    _run_pipeline(in_path=in_path, out_clean=out_clean, out_train=out_train, out_test=out_test, out_art=out_art)
    return {"train_path": out_train, "test_path": out_test, "artifacts_path": out_art}

def main() -> None:
    in_path = os.getenv("SALES_PATH", "data/raw/sales_raw.csv")
    out_clean = "data/raw/clean_sales.csv"
    out_train = "data/processed/train.csv"
    out_test = "data/processed/test.csv"
    out_art = "data/processed/feature_artifacts.json"

    _run_pipeline(in_path=in_path, out_clean=out_clean, out_train=out_train, out_test=out_test, out_art=out_art)



if __name__ == "__main__":
    main()
