"""Adapter implementing the ModelRegistryPort over MLflow."""
from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings
from ml_sentinel.domain.ports import ModelRegistryPort

logger = get_logger(__name__)


class MLflowRegistry(ModelRegistryPort):
    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()
        self.settings = settings

    def load_model(self, name: str, stage: str) -> Any:
        uri = f"models:/{name}/{stage}"
        logger.info("mlflow_load_model", uri=uri)
        return mlflow.pyfunc.load_model(model_uri=uri)

    def register_model(self, model: Any, name: str, **kwargs) -> str:
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=name, **kwargs)
        return name

    def transition_stage(self, name: str, version: str, stage: str) -> None:
        self.client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=True
        )
        logger.info("mlflow_stage_transition", name=name, version=version, stage=stage)

    def latest_versions(self, name: str, stage: str) -> list[dict]:
        versions = self.client.get_latest_versions(name, stages=[stage])
        return [{"version": v.version, "run_id": v.run_id, "stage": v.current_stage} for v in versions]
