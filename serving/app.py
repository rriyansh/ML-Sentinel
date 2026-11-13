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
    BatchPrediction
