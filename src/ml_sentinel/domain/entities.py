"""Domain entities (framework-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Prediction:
    value: float
    model_name: str
    model_version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class DriftResult:
    detected: bool
    drift_score: float
    threshold: float
    features: list[str]
    details: dict[str, Any]
    report_path: str | None = None


@dataclass(frozen=True)
class ModelMetrics:
    f1: float
    precision: float
    recall: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {"f1": self.f1, "precision": self.precision, "recall": self.recall, **self.extra}


@dataclass(frozen=True)
class FairnessResult:
    attribute: str
    demographic_parity_diff: float
    equalized_odds_diff: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
