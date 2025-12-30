from __future__ import annotations

import json
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _load_threshold_from_run(model_uri: str) -> float:
    """
    Reads model_config.json from the same MLflow run as model_uri.
    model_uri format: runs:/<run_id>/model
    """
    if not model_uri.startswith("runs:/"):
        raise ValueError(f"Unsupported model_uri: {model_uri}")

    run_id = model_uri.split("/")[1]
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "model_config.json")
    with open(local_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return float(cfg["threshold"])


def evaluate_model(model_uri: str, data_paths: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate model on test split, using the stored threshold.

    Returns a metrics dict that is safe to log.
    """
    test_df = pd.read_csv(data_paths["test_path"])
    # Rebuild the exact feature columns order from the run config
    run_id = model_uri.split("/")[1]
    client = mlflow.tracking.MlflowClient()
    cfg_path = client.download_artifacts(run_id, "model_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    feature_cols = list(cfg["feature_cols"])
    threshold = float(cfg["threshold"])

    # We need the same lag/rolling features as training
    from .train import build_features

    # Note: build_features expects train+test to compute lags; pass train from disk as well.
    train_df = pd.read_csv(data_paths["train_path"])
    train_fe, test_fe, _ = build_features(train_df, test_df)

    y_true = (test_fe["y_class"].to_numpy().astype(int) == 0).astype(int)
    X_test = test_fe[feature_cols]

    model = mlflow.xgboost.load_model(model_uri)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Save local metrics.json for downstream steps if needed
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
