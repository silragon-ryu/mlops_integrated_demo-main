"""Prefect flow orchestrating the sales quantity classification pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from prefect import flow, get_run_logger, task

from . import contracts
from .validate_inputs import validate_processed_datasets


@task(retries=2, retry_delay_seconds=30)
def preprocess_task(dummy_mode: bool = False) -> Dict[str, Any]:
    """Run feature engineering to produce processed datasets."""
    logger = get_run_logger()

    if dummy_mode:
        logger.info("Dummy mode enabled; skipping feature engineering.")
        return {
            "train_path": str(contracts.TRAIN_PROCESSED_PATH),
            "test_path": str(contracts.TEST_PROCESSED_PATH),
        }

    logger.info("Starting feature engineering step.")
    from src import feature_engineering

    data_paths = feature_engineering.run_feature_engineering()
    logger.info("Feature engineering completed: %s", data_paths)
    return data_paths


@task(retries=2, retry_delay_seconds=30)
def preflight_task(data_paths: Dict[str, Any], dummy_mode: bool = False) -> None:
    """Validate processed datasets and required columns."""
    logger = get_run_logger()

    if dummy_mode:
        logger.info("Dummy mode enabled; skipping preflight validation.")
        return

    train_path = Path(data_paths.get("train_path", contracts.TRAIN_PROCESSED_PATH))
    test_path = Path(data_paths.get("test_path", contracts.TEST_PROCESSED_PATH))

    logger.info("Running preflight validation for processed datasets.")
    validate_processed_datasets(
        train_path=train_path,
        test_path=test_path,
        required_columns=contracts.REQUIRED_PROCESSED_COLUMNS,
    )
    logger.info("Preflight validation passed.")


@task(retries=2, retry_delay_seconds=30)
def train_task(data_paths: Dict[str, Any], dummy_mode: bool = False) -> str:
    """Train the model using processed datasets."""
    logger = get_run_logger()

    if dummy_mode:
        logger.info("Dummy mode enabled; skipping training.")
        return "dummy-model-uri"

    logger.info("Starting model training.")
    from src import train

    model_uri = train.train_model(data_paths)
    logger.info("Training completed. Model URI: %s", model_uri)
    return model_uri


@task(retries=2, retry_delay_seconds=30)
def evaluate_task(model_uri: str, data_paths: Dict[str, Any], dummy_mode: bool = False) -> Dict[str, Any]:
    """Evaluate the trained model."""
    logger = get_run_logger()

    if dummy_mode:
        logger.info("Dummy mode enabled; skipping evaluation.")
        return {"accuracy": 0.0, "f1_macro": 0.0}

    logger.info("Starting model evaluation.")
    from src import evaluate

    metrics = evaluate.evaluate_model(model_uri=model_uri, data_paths=data_paths)
    logger.info("Evaluation completed: %s", metrics)
    return metrics


@task(retries=2, retry_delay_seconds=30)
def register_task(model_uri: str, metrics: Dict[str, Any], dummy_mode: bool = False) -> None:
    """Register the trained model in the MLflow model registry."""
    logger = get_run_logger()

    if dummy_mode:
        logger.info("Dummy mode enabled; skipping model registration.")
        return

    logger.info("Registering model %s to registry %s.", model_uri, contracts.MODEL_NAME)
    from src import register

    register.register_model(
        model_uri=model_uri,
        metrics=metrics,
        model_name=contracts.MODEL_NAME,
    )
    logger.info("Model registration completed.")


@flow(name="sales-quantity-mlops-pipeline")
def mlops_pipeline(dummy_mode: bool = True) -> Dict[str, Any]:
    """Unified Prefect pipeline for sales quantity classification."""
    logger = get_run_logger()
    logger.info("Launching MLOps pipeline. Dummy mode: %s", dummy_mode)

    data_paths_future = preprocess_task.submit(dummy_mode)
    data_paths = data_paths_future.result()

    preflight_future = preflight_task.submit(data_paths, dummy_mode)
    preflight_future.result()

    model_uri_future = train_task.submit(data_paths, dummy_mode, wait_for=[preflight_future])
    model_uri = model_uri_future.result()

    metrics_future = evaluate_task.submit(model_uri, data_paths, dummy_mode, wait_for=[model_uri_future])
    metrics = metrics_future.result()

    register_task.submit(model_uri, metrics, dummy_mode, wait_for=[metrics_future])

    logger.info("Pipeline completed successfully.")
    return {"model_uri": model_uri, "metrics": metrics, "data_paths": data_paths}


if __name__ == "__main__":
    mlops_pipeline()
