"""Pydantic v2 schemas for the serving API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TokenRequest(BaseModel):
    api_key: str = Field(..., min_length=8, description="Issued API key")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class PredictionRequest(BaseModel):
    conv_rate: float = Field(..., ge=0.0, le=1.0, description="Conversion rate")
    acc_rate: float = Field(..., ge=0.0, le=1.0, description="Acceptance rate")

    @field_validator("conv_rate", "acc_rate")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not (-1e9 < v < 1e9):
            raise ValueError("value out of finite range")
        return float(v)


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    predictions: list[float]
    model_name: str
    model_version: str
    request_id: str
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database_ok: bool
    details: dict[str, Any] = Field(default_factory=dict)
