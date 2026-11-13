"""
Structured logging with trace-context propagation.

- JSON logs via structlog (production) / pretty (dev)
- Automatically binds ``trace_id`` and ``span_id`` from OpenTelemetry
"""
from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path

import structlog
import yaml

from ml_sentinel.core.settings import get_settings

_CONFIG_CACHE: dict = {}
_DONE = False


def _load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE
    base = Path(__file__).resolve().parent.parent.parent.parent
    cfg = base / "config" / "logging.yaml"
    if cfg.exists():
        with cfg.open("r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    else:
        _CONFIG_CACHE = {
            "version": 1,
            "formatters": {"standard": {"format": "%(message)s"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }
    return _CONFIG_CACHE


def _otel_context_processor(_, __, event_dict: dict) -> dict:
    """Inject trace_id / span_id from the active OTel span."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:  # noqa: BLE001
        pass
    return event_dict


def configure_logging(force: bool = False) -> None:
    global _DONE
    if _DONE and not force:
        return
    settings = get_settings()
    logging.config.dictConfig(_load_config())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _otel_context_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _DONE = True


def get_logger(name: str = "ml_sentinel") -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name).bind(service="ml-sentinel")


def setup_logger(name: str = "ml_sentinel") -> structlog.stdlib.BoundLogger:
    return get_logger(name)
