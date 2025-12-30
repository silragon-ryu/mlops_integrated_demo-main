from __future__ import annotations

import hashlib
from typing import Dict
import pandas as pd


def stable_hash_to_bucket(value, num_buckets: int, salt: str = "") -> int:
    """Deterministic hash -> [0, num_buckets)."""
    if not isinstance(num_buckets, int) or num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive int, got {num_buckets}")

    s = (salt + str(value)).encode("utf-8")
    return int(hashlib.md5(s).hexdigest(), 16) % num_buckets


def add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add basic time features from a date column."""
    if date_col not in df.columns:
        raise ValueError(f"Missing column: {date_col}")

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        before = len(out)
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col])
        if len(out) < before:
            print(f"⚠️ Dropped {before - len(out)} invalid dates")

    out["year"] = out[date_col].dt.year.astype(int)
    out["month"] = out[date_col].dt.month.astype(int)
    out["day"] = out[date_col].dt.day.astype(int)
    out["dayofweek"] = out[date_col].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    return out


def add_hashed_features(
    df: pd.DataFrame,
    buckets: Dict[str, int],
    item_col: str = "ItemCode",
    branch_col: str = "BranchID",
    invoice_col: str = "InvoiceNumber",
) -> pd.DataFrame:
    """Add hashed ID features and simple crosses."""
    required = ["item", "branch", "invoice", "item_branch", "item_month"]
    for k in required:
        if k not in buckets or buckets[k] <= 0:
            raise ValueError(f"Invalid bucket for '{k}'")

    if "month" not in df.columns:
        raise ValueError("Run add_date_features() before hashing")

    out = df.copy()

    out[item_col] = out[item_col].astype(str)
    out[branch_col] = out[branch_col].astype(str)
    out[invoice_col] = out[invoice_col].astype(str)

    out["h_item"] = out[item_col].map(lambda x: stable_hash_to_bucket(x, buckets["item"], "item:"))
    out["h_branch"] = out[branch_col].map(lambda x: stable_hash_to_bucket(x, buckets["branch"], "branch:"))
    out["h_invoice"] = out[invoice_col].map(lambda x: stable_hash_to_bucket(x, buckets["invoice"], "inv:"))

    out["h_item_branch"] = (
        out[item_col].str.cat(out[branch_col], sep="|")
        .map(lambda x: stable_hash_to_bucket(x, buckets["item_branch"], "ib:"))
    )

    out["h_item_month"] = (
        out[item_col].str.cat(out["month"].astype(str), sep="|")
        .map(lambda x: stable_hash_to_bucket(x, buckets["item_month"], "im:"))
    )

    return out
