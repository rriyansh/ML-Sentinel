"""SQLAlchemy ORM models for audit and observability."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ml_sentinel.infrastructure.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DriftEvent(Base):
    __tablename__ = "drift_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    detected: Mapped[bool] = mapped_column(nullable=False)
    drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    features_monitored: Mapped[dict] = mapped_column(JSON, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ModelPromotion(Base):
    __tablename__ = "model_promotions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    promoted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    new_version: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    approved: Mapped[bool] = mapped_column(nullable=False)
    challenger_metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    champion_metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    policy: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    inputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    outputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="success")


class FairnessAudit(Base):
    __tablename__ = "fairness_audits"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    attribute: Mapped[str] = mapped_column(String(64), nullable=False)
    demographic_parity_diff: Mapped[float] = mapped_column(Float, nullable=False)
    equalized_odds_diff: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
