"""
ML-Sentinel FastAPI serving layer (v3.0) — async, hardened, traced.

Endpoints:
  POST /auth/token         — issue JWT from API key
  POST /auth/refresh       — rotate refresh token
  GET  /health/*           — k8s probes
  GET  /metrics            — Prometheus
  POST /api/v1/predict     — single prediction (scope: predict)
  POST /api/v1/predict/batch
  POST /api/v1/explain     — SHAP explanation (scope: explain)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ml_sentinel.core.circuit_breaker import CircuitBreaker
from ml_sentinel.core.errors import CircuitOpenError, ModelLoadError
from ml_sentinel.core.feature_flags import is_enabled
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import (
    MODEL_LOADED, record_prediction, set_model_ready,
)
from ml_sentinel.core.security import (
    create_access_token, create_refresh_token, decode_access_token, verify_api_key,
)
from ml_sentinel.core.settings import get_settings
from ml_sentinel.core.tracing import setup_tracing
from ml_sentinel.core.utils import timer
from ml_sentinel.infrastructure.database import ping_db
from monitoring.explainability import explain_prediction
from serving.dependencies import rate_limit, require_auth, require_scope
from serving.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from serving.schemas import (
    BatchPredictionRequest, ExplainRequest, ExplainResponse, HealthResponse,
    PredictionRequest, PredictionResponse, RefreshRequest, TokenRequest, TokenResponse,
)

logger = get_logger("serving.app")
settings = get_settings()

_state: dict[str, Any] = {"model": None, "name": "", "version": "", "stage": ""}
_breaker = CircuitBreaker(
    "model_predict",
    failure_threshold=settings.circuit_breaker_failure_threshold,
    recovery_seconds=settings.circuit_breaker_recovery_seconds,
)
_background: pd.DataFrame | None = None


def _load_model_from_registry() -> None:
    try:
        uri = f"models:/{settings.mlflow_model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri=uri)
        versions = mlflow.tracking.MlflowClient().get_latest_versions(
            settings.mlflow_model_name, stages=["Production"]
        )
        ver = versions[0].version if versions else "unknown"
        _state.update(model=model, name=settings.mlflow_model_name, version=ver, stage="Production")
        MODEL_LOADED.set(1)
        set_model_ready(True, name=settings.mlflow_model_name, version=ver, stage="Production")
        logger.info("model_loaded", name=settings.mlflow_model_name, version=ver)
    except Exception as exc:  # noqa: BLE001
        MODEL_LOADED.set(0)
        logger.exception("model_load_failed", error=str(exc))
        raise ModelLoadError(str(exc)) from exc


def _load_shap_background() -> None:
    global _background
    try:
        from pathlib import Path

        p = Path("data/reference/driver_stats.parquet")
        if p.exists():
            df = pd.read_parquet(p)[["conv_rate", "acc_rate"]]
            _background = df.sample(min(settings.shap_background_samples, len(df)), random_state=42)
            logger.info("shap_background_loaded", rows=len(_background))
    except Exception:  # noqa: BLE001
        logger.exception("shap_background_failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    setup_tracing()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_load_model_from_registry),
            timeout=settings.model_load_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("model_load_timeout")
    except ModelLoadError:
        logger.warning("starting_without_model")
    if is_enabled("enable_explain_endpoint"):
        await asyncio.to_thread(_load_shap_background)
    yield
    logger.info("shutdown_complete")


app = FastAPI(
    title="ML-Sentinel Serving API",
    version=settings.app_version,
    description="Enterprise model serving with drift-aware retraining, explainability, fairness.",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key", "X-Request-ID", "Content-Type"],
    max_age=600,
)


# ── Auth ──────────────────────────────────────────────────────
@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def issue_token(payload: TokenRequest) -> TokenResponse:
    if not verify_api_key(payload.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return TokenResponse(
        access_token=create_access_token(subject="api-key-user", scopes=["predict", "explain"]),
        expires_in=settings.jwt_expire_minutes * 60,
        refresh_token=create_refresh_token("api-key-user"),
    )


@app.post("/auth/refresh", response_model=TokenResponse, tags=["auth"])
def refresh_token(payload: RefreshRequest) -> TokenResponse:
    try:
        claims = decode_access_token(payload.refresh_token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(401, detail=str(exc)) from exc
    return TokenResponse(
        access_token=create_access_token(subject=claims["sub"], scopes=["predict", "explain"]),
        expires_in=settings.jwt_expire_minutes * 60,
    )


# ── Health ────────────────────────────────────────────────────
@app.get("/health/liveness", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/startup", tags=["health"])
def startup() -> dict[str, str]:
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded yet")
    return {"status": "started"}


@app.get("/health/readiness", response_model=HealthResponse, tags=["health"])
def readiness() -> HealthResponse:
    db_ok = ping_db()
    model_ok = _state["model"] is not None
    if not (db_ok and model_ok):
        raise HTTPException(503, detail={"database_ok": db_ok, "model_loaded": model_ok})
    return HealthResponse(
        status="ready", version=settings.app_version,
        model_loaded=True, database_ok=True,
        details={"model_version": _state["version"], "stage": _state["stage"]},
    )


# ── Metrics ───────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Prediction ────────────────────────────────────────────────
def _do_predict(records: list[list[float]]) -> list[float]:
    model = _state["model"]
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        out = _breaker.call(model.predict, records)
    except CircuitOpenError as exc:
        raise HTTPException(503, str(exc)) from exc
    return [float(x) for x in out]


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["predict"],
          dependencies=[Depends(rate_limit), Depends(require_auth), Depends(require_scope("predict"))])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    with timer() as elapsed:
        try:
            predictions = _do_predict([[payload.conv_rate, payload.acc_rate]])
            status_str = "success"
        except Exception as exc:  # noqa: BLE001
            record_prediction("error", elapsed())
            logger.exception("prediction_failed",
                             request_id=getattr(request.state, "request_id", "-"))
            raise HTTPException(500, f"Prediction failed: {exc}") from exc
    latency_ms = elapsed() * 1000
    record_prediction(status_str, elapsed())
    return PredictionResponse(
        predictions=predictions, model_name=_state["name"],
        model_version=_state["version"],
        request_id=getattr(request.state, "request_id", ""),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/api/v1/predict/batch", response_model=PredictionResponse, tags=["predict"],
          dependencies=[Depends(rate_limit), Depends(require_auth), Depends(require_scope("predict"))])
def predict_batch(payload: BatchPredictionRequest, request: Request) -> PredictionResponse:
    records = [[r.conv_rate, r.acc_rate] for r in payload.records]
    with timer() as elapsed:
        try:
            predictions = _do_predict(records)
            status_str = "success"
        except Exception as exc:  # noqa: BLE001
            record_prediction("error", elapsed())
            raise HTTPException(500, str(exc)) from exc
    latency_ms = elapsed() * 1000
    record_prediction(status_str, elapsed())
    return PredictionResponse(
        predictions=predictions, model_name=_state["name"],
        model_version=_state["version"],
        request_id=getattr(request.state, "request_id", ""),
        latency_ms=round(latency_ms, 2),
        metadata={"batch_size": len(records)},
    )


@app.post("/api/v1/explain", response_model=ExplainResponse, tags=["explain"],
          dependencies=[Depends(rate_limit), Depends(require_auth), Depends(require_scope("explain"))])
def explain(payload: ExplainRequest) -> ExplainResponse:
    if not is_enabled("enable_explain_endpoint", default=True):
        raise HTTPException(404, "Explainability disabled")
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded")
    if _background is None:
        raise HTTPException(503, "SHAP background not available")

    instance = pd.DataFrame([{"conv_rate": payload.conv_rate, "acc_rate": payload.acc_rate}])
    try:
        result = explain_prediction(_state["model"], _background, instance, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Explain failed: {exc}") from exc

    pred = _do_predict([[payload.conv_rate, payload.acc_rate]])[0]
    return ExplainResponse(
        contributions=result["contributions"],
        base_value=result["base_value"],
        prediction=pred,
        model_version=_state["version"],
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception",
                     request_id=getattr(request.state, "request_id", "-"),
                     error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error",
                 "request_id": getattr(request.state, "request_id", "")},
    )
