"""
Prometheus metrics for the serving layer.

Metrics are registered once at import time. FastAPI middleware and
endpoint code update them via the ``record_*`` helpers.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, REGISTRY


def _safe_counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    try:
        return Counter(name, doc, labels or [])
    except ValueError:  # already registered (e.g. during reload)
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


def _safe_histogram(
    name: str, doc: str, labels: list[str] | None = None, buckets: list[float] | None = None
) -> Histogram:
    try:
        return Histogram(name, doc, labels or [], buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


def _safe_gauge(name: str, doc: str, labels: list[str] | None = None) -> Gauge:
    try:
        return Gauge(name, doc, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


# ── Metrics definitions ────────────────────────────────────────────
HTTP_REQUESTS = _safe_counter(
    "ml_sentinel_http_requests_total",
    "Total HTTP requests by method, endpoint, status.",
    ["method", "endpoint", "status"],
)
HTTP_LATENCY = _safe_histogram(
    "ml_sentinel_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTIONS = _safe_counter(
    "ml_sentinel_predictions_total",
    "Total predictions by status.",
    ["status"],
)
PREDICTION_LATENCY = _safe_histogram(
    "ml_sentinel_prediction_latency_seconds",
    "Prediction latency in seconds.",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_LOADED = _safe_gauge(
    "ml_sentinel_model_loaded",
    "1 if a model is loaded and ready, else 0.",
)
MODEL_INFO = _safe_gauge(
    "ml_sentinel_model_info",
    "Model version information.",
    ["model_name", "model_version", "stage"],
)
DRIFT_EVENTS = _safe_counter(
    "ml_sentinel_drift_events_total",
    "Number of drift checks by outcome.",
    ["detected"],
)
RATE_LIMITED = _safe_counter(
    "ml_sentinel_rate_limited_total",
    "Number of requests rejected by the rate limiter.",
    ["endpoint"],
)


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
