from __future__ import annotations

from typing import Any, Dict

import mlflow


def register_model(model_uri: str, metrics: Dict[str, Any], model_name: str = "sales-quantity-classifier") -> None:
    """
    Registers the model in MLflow Model Registry and sets alias "@production".

    For a demo: we register every successful run as a new version and point the alias to it.
    """
    client = mlflow.tracking.MlflowClient()

    # Register
    mv = mlflow.register_model(model_uri, model_name)

    # Optional: add a few metric tags for quick browsing
    try:
        client.set_model_version_tag(model_name, mv.version, "f1", str(metrics.get("f1")))
        client.set_model_version_tag(model_name, mv.version, "accuracy", str(metrics.get("accuracy")))
    except Exception:
        pass

    # Alias: production
    client.set_registered_model_alias(model_name, "@production", mv.version)
