"""Input validation utilities for the Prefect pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import pandas as pd


class ValidationError(Exception):
    """Raised when pipeline input validation fails."""


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")


def _validate_columns(path: Path, required_columns: Set[str]) -> None:
    df_head = pd.read_csv(path, nrows=0)
    missing = required_columns - set(df_head.columns)
    if missing:
        raise ValidationError(
            f"File {path} is missing required columns: {sorted(missing)}"
        )


def validate_processed_datasets(
    *,
    train_path: Path,
    test_path: Path,
    required_columns: Iterable[str],
) -> None:
    """Validate that processed datasets exist and contain required columns.

    Args:
        train_path: Path to the processed training CSV.
        test_path: Path to the processed test CSV.
        required_columns: Columns that must be present in both datasets.

    Raises:
        FileNotFoundError: If any expected dataset file does not exist.
        ValidationError: If a dataset is invalid or missing required columns.
    """

    required_set = set(required_columns)

    for path in (train_path, test_path):
        _ensure_file_exists(path)
        _validate_columns(path, required_set)
