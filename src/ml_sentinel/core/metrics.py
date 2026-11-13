"""Prometheus metrics (multi-process safe)."""
from __future__ import annotations

import os

from prometheus_client import Counter, Gauge, Histogram, REGISTRY

if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    # Enable multiprocess mode
    try:
        from prometheus_client import multiprocess  # noqa: F401
    except ImportError:
        pass


def _safe(cls, name, doc, labels=None, **kw):
    try:
        return cls(name, doc, labels or [], **kw)
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


HTTP_REQUESTS = _safe(Counter, "ml_sentinel_http_requests_total",
                     "HTTP requests", ["method", "endpoint", "status"])
HTTP_LATENCY = _safe(Histogram, "ml_sentinel_http_request_duration_seconds",
                     "HTTP latency", ["method", "endpoint"],
                     buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5])
PREDICTIONS = _safe(Counter, "ml_sentinel_predictions_total", "Predictions", ["status"])
PREDICTION_LATENCY = _safe(Histogram, "ml_sentinel_prediction_latency_seconds",
                           "Prediction latency", buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
MODEL_LOADED = _safe(Gauge, "ml_sentinel_model_loaded", "Model ready flag")
MODEL_INFO = _safe(Gauge, "ml_sentinel_model_info", "Model metadata",
                   ["model_name", "model_version", "stage"])
DRIFT_EVENTS = _safe(Counter, "ml_sentinel_drift_events_total", "Drift events", ["detected"])
RATE_LIMITED = _safe(Counter, "ml_sentinel_rate_limited_total", "Rate-limited", ["endpoint"])
AUTH_FAILURES = _safe(Counter, "ml_sentinel_auth_failures_total", "Auth failures", ["reason"])
CIRCUIT_STATE = _safe(Gauge, "ml_sentinel_circuit_state", "Circuit breaker state (0=closed,1=half,2=open)", ["dependency"])
EXPLANATIONS = _safe(Counter, "ml_sentinel_explanations_total", "SHAP explanations", ["status"])
FAIRNESS_RUNS = _safe(Counter, "ml_sentinel_fairness_runs_total", "Fairness checks", ["passed"])


def record_http(method: str, endpoint: str, status: int, latency_s: float) -> None:
    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, endpoint=endpoint).observe(latency_s)


def record_prediction(status: str, latency_s: float) -> None:
    PREDICTIONS.labels(status=status).inc()
    PREDICTION_LATENCY.observe(latency_s)


def record_drift(detected: bool) -> None:
    DRIFT_EVENTS.labels(detected=str(detected).lower()).inc()


def set_model_ready(ready: bool, *, name: str = "", version: str = "", stage: str = "") -> None:
    MODEL_LOADED.set(1 if ready else 0)
    if ready:
        MODEL_INFO.labels(model_name=name, model_version=version, stage=stage).set(1)
