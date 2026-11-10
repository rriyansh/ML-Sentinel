"""
Centralised, validated configuration for ML-Sentinel.

All values come from environment variables (optionally loaded from .env).
In ``ENVIRONMENT=production`` the settings module **fails fast** if any
required secret is missing or weak, which protects against accidental
deployments with development defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=False,
    )

    # ── Application ────────────────────────────────────────────────
    app_name: str = "ML-Sentinel"
    app_version: str = "2.0.0"
    environment: Environment = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Security ───────────────────────────────────────────────────
    secret_key: SecretStr = Field(
        default=SecretStr("dev-insecure-secret-change-me-please-1234"),
        description="Signing key for JWT. Must be ≥32 chars in production.",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_keys: str = "dev-key-abc123"
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"
    rate_limit_per_minute: int = 60

    # ── MLflow ─────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "ml_sentinel_production"
    mlflow_model_name: str = "MLSentinelModel"

    # ── Feast ──────────────────────────────────────────────────────
    feast_repo_path: str = "feast_repo"
    feast_tracking_uri: str = "http://mlflow:5000"

    # ── Database ───────────────────────────────────────────────────
    database_url: str = (
        "postgresql+psycopg2://mlsentinel:mlsentinel@postgres:5432/mlsentinel"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # ── Redis ──────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"

    # ── Serving ────────────────────────────────────────────────────
    serving_host: str = "0.0.0.0"
    serving_port: int = 8000
    model_load_timeout: int = 30
    prediction_timeout: int = 10

    # ── Monitoring ─────────────────────────────────────────────────
    drift_report_dir: str = "monitoring/reports"
    drift_alert_threshold: float = 0.4

    # ── Validators ─────────────────────────────────────────────────
    @field_validator("secret_key")
    @classmethod
    def _validate_secret(cls, v: SecretStr) -> SecretStr:
        if v.get_secret_value() in {"", "changeme", "secret"}:
            raise ValueError("SECRET_KEY must not be a well-known placeholder")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in levels:
            raise ValueError(f"LOG_LEVEL must be one of {levels}")
        return v.upper()

    @model_validator(mode="after")
    def _validate_production(self) -> "Settings":
        if self.environment != "production":
            return self

        errors: list[str] = []

        if self.debug:
            errors.append("DEBUG must be false in production")

        if len(self.secret_key.get_secret_value()) < 32:
            errors.append("SECRET_KEY must be ≥32 chars in production")

        if self.default_api_key() in {"dev-key-abc123", ""}:
            errors.append("API_KEYS must be rotated from the dev default in production")

        if not self.allowed_origins or "*" in self.allowed_origins:
            errors.append("ALLOWED_ORIGINS must be an explicit allow-list in production")

        if errors:
            raise ValueError("Production configuration invalid:\n  - " + "\n  - ".join(errors))
        return self

    # ── Convenience accessors ──────────────────────────────────────
    def api_key_set(self) -> set[str]:
        """Return the set of accepted API keys (used during key rotation)."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    def default_api_key(self) -> str:
        keys = self.api_key_set()
        return next(iter(keys)) if keys else ""

    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance."""
    return Settings()


# Convenient module-level handle; import ``settings`` directly for most uses.
settings = get_settings()


def reload_settings() -> Settings:
    """Clear the cache and reload settings — useful in tests."""
    os.environ.pop("_SETTINGS_LOADED", None)
    get_settings.cache_clear()
    return get_settings()
