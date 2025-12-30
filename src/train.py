from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import mlflow

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise RuntimeError("xgboost is required for training. Install requirements.txt") from e

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


# =========================
# Config
# =========================
@dataclass
class TrainConfig:
    model_name: str = "sales-quantity-classifier"
    experiment_name: str = "sales-quantity-demo"
    tracking_uri: str | None = None  # e.g. http://localhost:5000
    random_seed: int = 42

    # Feature generation
    lags: Tuple[int, ...] = (1, 2, 3, 7, 14)
    windows: Tuple[int, ...] = (3, 7, 14, 30)

    # CV + thresholding
    n_splits: int = 5
    threshold_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))

    # XGBoost params
    params: Dict[str, Any] | None = None


def default_xgb_params(seed: int) -> Dict[str, Any]:
    """CPU-friendly defaults; deterministic via random_state."""
    return dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
    )


def ensure_config(cfg: TrainConfig) -> TrainConfig:
    if cfg.params is None:
        cfg.params = default_xgb_params(cfg.random_seed)
    else:
        cfg.params.setdefault("random_state", cfg.random_seed)
    return cfg


# =========================
# Features
# =========================
REQUIRED_COLS = {"year", "month", "day", "QuantitySold", "y_class", "h_item_branch"}


def validate_required_columns(df: pd.DataFrame, name: str) -> None:
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def make_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")


def add_lag_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    lags: Tuple[int, ...],
    windows: Tuple[int, ...],
) -> pd.DataFrame:
    """Past-only lag and rolling features derived from QuantitySold."""
    out = df.copy()
    out["__date"] = make_date(out)
    out = out.sort_values([group_col, "__date"]).reset_index(drop=True)

    g = out.groupby(group_col, sort=False)["QuantitySold"]

    for k in lags:
        out[f"qty_lag_{k}"] = g.shift(k)

    for w in windows:
        shifted = g.shift(1)  # past-only
        out[f"qty_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        out[f"qty_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0.0)
        out[f"qty_roll_min_{w}"] = shifted.rolling(w, min_periods=1).min()
        out[f"qty_roll_max_{w}"] = shifted.rolling(w, min_periods=1).max()

    out = out.drop(columns=["__date"])
    return out


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: TrainConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Build lag features using train+test history, then split back."""
    combined = pd.concat(
        [train_df.assign(__split="train"), test_df.assign(__split="test")],
        axis=0,
        ignore_index=True,
    )

    combined = add_lag_rolling_features(
        combined,
        group_col="h_item_branch",
        lags=cfg.lags,
        windows=cfg.windows,
    )

    lag_cols = [c for c in combined.columns if c.startswith("qty_")]
    combined[lag_cols] = combined[lag_cols].fillna(0.0)

    drop_cols = {"QuantitySold", "y_class", "__split"}
    feature_cols = [c for c in combined.columns if c not in drop_cols]

    train_out = combined[combined["__split"] == "train"].drop(columns=["__split"]).reset_index(drop=True)
    test_out = combined[combined["__split"] == "test"].drop(columns=["__split"]).reset_index(drop=True)
    return train_out, test_out, feature_cols


# =========================
# CV / Threshold sweep
# =========================
def threshold_sweep_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    thresholds: List[float],
    n_splits: int,
    seed: int,
) -> Tuple[float, float, Dict[float, float], Dict[float, List[float]]]:
    """Return best threshold, best mean F1, mean score per threshold, and fold scores."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores_by_t: Dict[float, List[float]] = {t: [] for t in thresholds}

    params = default_xgb_params(seed)

    for _, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)

        proba = m.predict_proba(X_va)[:, 1]
        for t in thresholds:
            pred = (proba >= t).astype(int)
            fold_scores_by_t[t].append(float(f1_score(y_va, pred)))

    mean_scores = {t: float(np.mean(v)) for t, v in fold_scores_by_t.items()}
    best_t = max(mean_scores, key=mean_scores.get)
    best_mean_f1 = float(mean_scores[best_t])

    return float(best_t), best_mean_f1, mean_scores, fold_scores_by_t


# =========================
# Artifact logging (reliable)
# =========================
def log_proof_artifact() -> None:
    """Always logs a single proof file so you know artifacts worked for this run."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "PROOF.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("artifact upload works for THIS training run\n")
        mlflow.log_artifact(p)


def log_artifacts_individually(
    model: xgb.XGBClassifier,
    model_config: Dict[str, Any],
    X_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    """
    Logs each artifact file individually under artifacts_bundle/.
    This is more reliable than mlflow.log_artifacts(dir) across setups.
    """
    with tempfile.TemporaryDirectory() as td:
        paths: Dict[str, str] = {}

        paths["xgb_model.json"] = os.path.join(td, "xgb_model.json")
        model.save_model(paths["xgb_model.json"])

        paths["model_config.json"] = os.path.join(td, "model_config.json")
        with open(paths["model_config.json"], "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        paths["input_sample.csv"] = os.path.join(td, "input_sample.csv")
        X_sample.to_csv(paths["input_sample.csv"], index=False)

        paths["feature_importance.csv"] = os.path.join(td, "feature_importance.csv")
        feature_importance.to_csv(paths["feature_importance.csv"], index=False)

        paths["ARTIFACT_OK.txt"] = os.path.join(td, "ARTIFACT_OK.txt")
        with open(paths["ARTIFACT_OK.txt"], "w", encoding="utf-8") as f:
            f.write("If you see this, artifacts_bundle upload worked.\n")

        for _, path in paths.items():
            mlflow.log_artifact(path, artifact_path="artifacts_bundle")


def log_prebuilt_artifacts(prebuilt_dir: str) -> None:
    """
    Option A: Upload an existing local folder (e.g., model_artifact/) into MLflow.
    Uses per-file logging for reliability and preserves subfolders.
    Appears under: prebuilt_model_artifact/
    """
    if not prebuilt_dir or not os.path.isdir(prebuilt_dir):
        return

    for root, _, files in os.walk(prebuilt_dir):
        for fname in files:
            src_path = os.path.join(root, fname)

            rel_dir = os.path.relpath(root, prebuilt_dir)
            artifact_subdir = (
                "prebuilt_model_artifact"
                if rel_dir in (".", "")
                else os.path.join("prebuilt_model_artifact", rel_dir)
            )

            mlflow.log_artifact(src_path, artifact_path=artifact_subdir)


# =========================
# Training
# =========================
def train_model(
    data_paths: Dict[str, Any],
    cfg: TrainConfig | None = None,
    prebuilt_artifact_dir: str = "model_artifact",
) -> str:
    cfg = ensure_config(cfg or TrainConfig())

    tracking_uri = cfg.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set and cfg.tracking_uri is None")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    train_path = data_paths["train_path"]
    test_path = data_paths["test_path"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    validate_required_columns(train_df, "train.csv")
    validate_required_columns(test_df, "test.csv")

    train_fe, test_fe, feature_cols = build_features(train_df, test_df, cfg)

    # LOW vs REST
    y = (train_fe["y_class"].to_numpy().astype(int) == 0).astype(int)
    groups = train_fe["h_item_branch"].to_numpy()

    X = train_fe[feature_cols]
    X_test = test_fe[feature_cols]

    thresholds = list(cfg.threshold_grid)

    best_t, best_mean_f1, mean_scores, fold_scores = threshold_sweep_cv(
        X=X,
        y=y,
        groups=groups,
        thresholds=thresholds,
        n_splits=cfg.n_splits,
        seed=cfg.random_seed,
    )

    model = xgb.XGBClassifier(**cfg.params)
    model.fit(X, y)

    with mlflow.start_run(run_name="xgb_low_vs_rest") as run:
        # Proof artifact first
        try:
            log_proof_artifact()
        except Exception as e:
            mlflow.set_tag("proof_artifact_error", str(e))

        # Upload your existing model_artifact folder (Option A)
        try:
            log_prebuilt_artifacts(prebuilt_artifact_dir)
        except Exception as e:
            mlflow.set_tag("prebuilt_artifact_error", str(e))

        # Params
        mlflow.log_params({f"xgb_{k}": v for k, v in cfg.params.items()})
        mlflow.log_param("best_threshold", best_t)
        mlflow.log_param("cv_best_mean_f1", best_mean_f1)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("n_splits", cfg.n_splits)
        mlflow.log_param("threshold_grid_size", len(thresholds))
        mlflow.log_param("train_path", str(train_path))
        mlflow.log_param("test_path", str(test_path))

        # Metrics: curve of mean CV F1 over thresholds
        for t, score in mean_scores.items():
            mlflow.log_metric("cv_mean_f1_by_threshold", float(score), step=int(round(t * 100)))

        # Metrics: fold variance at best threshold
        for fold_idx, f1v in enumerate(fold_scores[best_t]):
            mlflow.log_metric("cv_f1_best_threshold_by_fold", float(f1v), step=fold_idx)

        # Train metric
        train_proba = model.predict_proba(X)[:, 1]
        train_pred = (train_proba >= best_t).astype(int)
        mlflow.log_metric("train_f1", float(f1_score(y, train_pred)))

        # Sanity stats on test predictions
        try:
            test_proba = model.predict_proba(X_test)[:, 1]
            mlflow.log_metric("test_pred_mean_proba", float(np.mean(test_proba)))
            mlflow.log_metric("test_pred_std_proba", float(np.std(test_proba)))
        except Exception as e:
            mlflow.set_tag("test_pred_stats_error", str(e))

        # Artifacts bundle (individual files, reliable)
        model_config = {
            "model_name": cfg.model_name,
            "experiment_name": cfg.experiment_name,
            "run_id": run.info.run_id,
            "target": "low_vs_rest",
            "threshold": best_t,
            "group_col": "h_item_branch",
            "feature_cols": feature_cols,
            "model_file": "artifacts_bundle/xgb_model.json",
        }

        fi = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_.tolist()}
        ).sort_values("importance", ascending=False)

        try:
            log_artifacts_individually(
                model=model,
                model_config=model_config,
                X_sample=X.head(50),
                feature_importance=fi,
            )
        except Exception as e:
            mlflow.set_tag("artifact_logging_error", str(e))

        # Useful tags
        mlflow.set_tag("model_format", "xgboost")
        mlflow.set_tag("target", "low_vs_rest")
        mlflow.set_tag("tracking_uri", tracking_uri)

    return run.info.run_id


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and log rich metrics + reliable artifacts to MLflow."
    )
    p.add_argument("--train_path", required=True, help="Path to processed train.csv")
    p.add_argument("--test_path", required=True, help="Path to processed test.csv")
    p.add_argument("--tracking_uri", default=None, help="MLflow tracking URI (e.g., http://localhost:5000)")
    p.add_argument("--experiment", default="sales-quantity-demo", help="MLflow experiment name")
    p.add_argument("--model_name", default="sales-quantity-classifier", help="Logical model name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--prebuilt_artifact_dir",
        default="model_artifact",
        help="Existing folder of prebuilt artifacts to upload to MLflow.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        random_seed=args.seed,
    )

    run_id = train_model(
        {"train_path": args.train_path, "test_path": args.test_path},
        cfg=cfg,
        prebuilt_artifact_dir=args.prebuilt_artifact_dir,
    )
    print(f"Run logged successfully. run_id={run_id}")
