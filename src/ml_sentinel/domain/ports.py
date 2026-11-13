"""Ports (abstract interfaces) — implemented by infrastructure adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ml_sentinel.domain.entities import DriftResult, ModelMetrics


class FeatureStorePort(ABC):
    @abstractmethod
    def get_historical_features(
        self, entity_df: pd.DataFrame, features: list[str]
    ) -> pd.DataFrame: ...


class ModelRegistryPort(ABC):
    @abstractmethod
    def load_model(self, name: str, stage: str) -> Any: ...

    @abstractmethod
    def register_model(self, model: Any, name: str, **kwargs) -> str: ...

    @abstractmethod
    def transition_stage(self, name: str, version: str, stage: str) -> None: ...

    @abstractmethod
    def latest_versions(self, name: str, stage: str) -> list[dict]: ...


class DriftDetectorPort(ABC):
    @abstractmethod
    def detect(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult: ...


class MetricsRepositoryPort(ABC):
    @abstractmethod
    def save_metrics(self, run_id: str, metrics: ModelMetrics) -> None: ...

    @abstractmethod
    def get_champion_metrics(self, model_name: str) -> ModelMetrics | None: ...
