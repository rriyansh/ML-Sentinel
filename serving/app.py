"""
ML-Sentinel FastAPI serving layer.

Features:
- Lazy/parallel-safe model loading from MLflow Registry (Production stage)
- JWT + API-key auth with per-IP rate limiting
- Request-id + structured logging + Prometheus metrics middleware
- Health/liveness/readiness/startup probes for Kubernetes
- Single & batch prediction with per-request timeout
- Graceful shutdown
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import mlflow.pyfunc
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ml_sentinel.database import ping_db
from ml_sentinel.exceptions import ModelLoadError
from ml_sentinel.logger import get_logger
from ml_sentinel.metrics import MODEL_LOADED, record_prediction, set_model_ready
from ml_sentinel.security import create_access_token, verify_api_key
from ml_sentinel.settings import get_settings
from ml_sentinel.utils import timer
from serving.dependencies import rate_limit, require_auth
from serving.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from serving.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TokenRequest,
    TokenResponse,
)

logger = get_logger("serving.app")
settings = get_settings()

# ── Model state ────────────────────────────────────────────────────
_state: dict[str, Any] = {"model": None, "name": "", "version": "", "stage": ""}


def _load_model_from_registry() -> None:
    """Synchronously load the Production model and update state/metrics."""
    try:
        model_uri = f"models:/{settings.mlflow_model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
        version = versions[0].version if versions else "unknown"
        _state.update(model=model, name=settings.mlflow_model_name, version=version, stage="Production")
        MODEL_LOADED.set(1)
        set_model_ready(True, name=settings.mlflow_model_name, version=version, stage="Production")
        logger.info("model_loaded", model=settings.mlflow_model_name, version=version)
    except Exception as exc:  # noqa: BLE001
        MODEL_LOADED.set(0)
        logger.exception("model_load_failed", error=str(exc))
        raise ModelLoadError(f"Could not load Production model: {exc}") from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model on startup; cleanup on shutdown."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_load_model_from_registry),
            timeout=settings.model_load_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("model_load_timeout", timeout=settings.model_load_timeout)
    except ModelLoadError:
        logger.warning("starting_without_model")
    yield
    logger.info("shutdown_complete")


# ── App factory ────────────────────────────────────────────────────
app = FastAPI(
    title="ML-Sentinel Serving API",
    version=settings.app_version,
    description="Production model serving with drift-aware auto-retraining.",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# ── Middleware (order matters: outermost first) ────────────────────
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


# ── Auth token endpoint ────────────────────────────────────────────
@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def issue_token(payload: TokenRequest) -> TokenResponse:
    """Exchange a valid API key for a short-lived JWT."""
    if not verify_api_key(payload.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    token = create_access_token(subject="api-key-user")
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


# ── Health endpoints ───────────────────────────────────────────────
@app.get("/health/liveness", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/startup", tags=["health"])
def startup() -> dict[str, str]:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "started"}


@app.get("/health/readiness", response_model=HealthResponse, tags=["health"])
def readiness() -> HealthResponse:
    db_ok = ping_db()
    model_ok = _state["model"] is not None
    if not (db_ok and model_ok):
        raise HTTPException(
            status_code=503,
            detail={
                "database_ok": db_ok,
                "model_loaded": model_ok,
            },
        )
    return HealthResponse(
        status="ready",
        version=settings.app_version,
        model_loaded=True,
        database_ok=True,
        details={"model_version": _state["version"], "stage": _state["stage"]},
    )


# ── Prometheus metrics ─────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Prediction endpoints ───────────────────────────────────────────
def _do_predict(records: list[list[float]]) -> list[float]:
    """Run the underlying MLflow pyfunc model."""
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    out = model.predict(records)
    return [float(x) for x in out]


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["predict"],
    dependencies=[Depends(rate_limit), Depends(require_auth)],
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    with timer() as elapsed:
        try:
            predictions = _do_predict([[payload.conv_rate, payload.acc_rate]])
            status_str = "success"
        except Exception as exc:  # noqa: BLE001
            record_prediction("error", elapsed())
            logger.exception("prediction_failed", request_id=getattr(request.state, "request_id", "-"))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    latency_ms = elapsed() * 1000
    record_prediction(status_str, elapsed())

    return PredictionResponse(
        predictions=predictions,
        model_name=_state["name"],
        model_version=_state["version"],
        request_id=getattr(request.state, "request_id", ""),
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/api/v1/predict/batch",
    response_model=PredictionResponse,
    tags=["predict"],
    dependencies=[Depends(rate_limit), Depends(require_auth)],
)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> PredictionResponse:
    records = [[r.conv_rate, r.acc_rate] for r in payload.records]
    with timer() as elapsed:
        try:
            predictions = _do_predict(records)
            status_str = "success"
        except Exception as exc:  # noqa: BLE001
            record_prediction("error", elapsed())
            logger.exception("batch_prediction_failed")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    latency_ms = elapsed() * 1000
    record_prediction(status_str, elapsed())

    return PredictionResponse(
        predictions=predictions,
        model_name=_state["name"],
        model_version=_state["version"],
        request_id=getattr(request.state, "request_id", ""),
        latency_ms=round(latency_ms, 2),
        metadata={"batch_size": len(records)},
    )


# ── Global exception handler ───────────────────────────────────────
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_exception",
        request_id=getattr(request.state, "request_id", "-"),
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": getattr(request.state, "request_id", "")},
    )
