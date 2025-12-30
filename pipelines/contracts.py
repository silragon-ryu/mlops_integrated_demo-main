"""Contracts and constants for the Prefect MLOps pipeline.

This module centralizes shared configuration such as model registry names,
expected processed data locations, and required columns for the sales quantity
classification pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, Set

# Base directories
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

# Expected processed dataset paths
TRAIN_PROCESSED_PATH: Final[Path] = PROCESSED_DIR / "train.csv"
TEST_PROCESSED_PATH: Final[Path] = PROCESSED_DIR / "test.csv"

# MLflow model registry configuration
MODEL_NAME: Final[str] = "sales-quantity-classifier"

# Required columns that must exist in processed datasets
REQUIRED_PROCESSED_COLUMNS: Final[Set[str]] = {
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
}

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PROCESSED_DIR",
    "TRAIN_PROCESSED_PATH",
    "TEST_PROCESSED_PATH",
    "MODEL_NAME",
    "REQUIRED_PROCESSED_COLUMNS",
]
