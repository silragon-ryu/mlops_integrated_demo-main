from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_TITLE = "Sales Quantity Risk API (Low vs Rest)"
MODEL_NAME = os.getenv("MODEL_NAME", "sales-quantity-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "@production")
PROCESSED_TRAIN_PATH = os.getenv("TRAIN_PROCESSED_PATH", "data/processed/train.csv")


class RowInput(BaseModel):
    y_class: Optional[int] = Field(default=None, description="Optional, only for debugging")
    QuantitySold: float
    year: int
    month: int
    day: int
    dayofweek: int
    is_weekend: int
    h_item: int
    h_branch: int
    h_invoice: int
    h_item_branch: int
    h_item_month: int


class PredictRequest(BaseModel):
    rows: List[RowInput]


class PredictResponseItem(BaseModel):
    low_risk: int
    probability_low: float
    threshold: float


app = FastAPI(title=APP_TITLE)


@lru_cache(maxsize=1)
def _load_registry_cfg() -> Dict[str, Any]:
    client = mlflow.tracking.MlflowClient()
    # Get latest version behind alias
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = mv.run_id
    local_path = client.download_artifacts(run_id, "model_config.json")
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.xgboost.load_model(model_uri)


@lru_cache(maxsize=1)
def _load_train_history() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        raise FileNotFoundError(f"Missing train data at {PROCESSED_TRAIN_PATH}. Run feature engineering first.")
    return pd.read_csv(PROCESSED_TRAIN_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        cfg = _load_registry_cfg()
        return {"status": "ok", "model": cfg.get("model_name"), "threshold": cfg.get("threshold")}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        cfg = _load_registry_cfg()
        model = _load_model()
        threshold = float(cfg["threshold"])
        feature_cols = list(cfg["feature_cols"])

        # Incoming rows
        req_df = pd.DataFrame([r.model_dump() for r in req.rows])

        # Build lag/rolling features using train history as past context
        from src.train import build_features

        train_df = _load_train_history()
        _, req_fe, _ = build_features(train_df, req_df)

        X = req_fe[feature_cols]
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        return [
            PredictResponseItem(low_risk=int(p), probability_low=float(pr), threshold=threshold)
            for p, pr in zip(pred, proba)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
