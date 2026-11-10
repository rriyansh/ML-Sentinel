"""
Structured logging setup for ML-Sentinel.

Uses ``structlog`` for JSON-formatted logs suitable for ingestion into
Loki / ELK / Datadog. Falls back gracefully to plain text for CLI usage.
"""
from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path

import structlog
import yaml

from ml_sentinel.settings import get_settings


_CONFIG_CACHE: dict = {}


def _load_logging_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE

    base_dir = Path(__file__).resolve().parent.parent.parent
    cfg_path = base_dir / "config" / "logging.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
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


def configure_logging(force: bool = False) -> None:
    """Configure stdlib logging + structlog. Idempotent unless ``force``."""
    if getattr(configure_logging, "_done", False) and not force:
        return

    settings = get_settings()
    logging.config.dictConfig(_load_logging_config())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
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
    configure_logging._done = True  # type: ignore[attr-defined]


def get_logger(name: str = "ml_sentinel") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with the given name bound."""
    configure_logging()
    return structlog.get_logger(name).bind(service="ml-sentinel")


# Backwards-compatible alias used across modules
def setup_logger(name: str = "ml_sentinel") -> structlog.stdlib.BoundLogger:
    return get_logger(name)
