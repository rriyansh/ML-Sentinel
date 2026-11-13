"""
Centralized, validated configuration for ML-Sentinel v3.0.

Loads from environment / .env, with strict production validation.
Secrets are wrapped in ``SecretStr`` and never logged.
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]
SecretsBackend = Literal["env", "vault", "aws", "gcp"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────
    app_name: str = "ML-Sentinel"
    app_version: str = "3.0.0"
    environment: Environment = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Security ──────────────────────────────────────────
    secret_key: SecretStr = Field(
        default=SecretStr("dev-insecure-secret-change-me-please-1234"),
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    jwt_refresh_expire_days: int = 7
    api_keys: str = "dev-key-abc123"
    oidc_enabled: bool = False
    oidc_issuer_url: str = ""
    oidc_client_id: str = ""
    oidc_audience: str = ""
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10

    # ── Encryption ────────────────────────────────────────
    encryption_key: SecretStr = Field(
        default=SecretStr(
            base64.urlsafe_b64encode(b"0" * 32).decode()  # dev-only placeholder
        )
    )
    pii_fields: str = "driver_id"

    # ── Secrets backend ───────────────────────────────────
    secrets_backend: SecretsBackend = "env"
    vault_addr: str = ""
    vault_token: SecretStr = SecretStr("")
    vault_secret_path: str = "ml-sentinel"

    # ── MLflow ────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "ml_sentinel_production"
    mlflow_model_name: str = "MLSentinelModel"
    mlflow_enable_otel: bool = True

    # ── Feast ─────────────────────────────────────────────
    feast_repo_path: str = "feast_repo"
    feast_tracking_uri: str = "http://mlflow:5000"

    # ── Database ──────────────────────────────────────────
    database_url: str = "postgresql+psycopg2://mlsentinel:mlsentinel@postgres:5432/mlsentinel"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_statement_timeout_ms: int = 30000

    # ── Redis ─────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    redis_rate_limit_prefix: str = "mls:rl:"

    # ── Serving ───────────────────────────────────────────
    serving_host: str = "0.0.0.0"
    serving_port: int = 8000
    model_load_timeout: int = 30
    prediction_timeout: int = 10
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: int = 30

    # ── Observability ─────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "ml-sentinel-serving"
    otel_traces_sampler_arg: float = 0.1
    prometheus_multiproc_dir: str = "/tmp/prom"

    # ── Monitoring ────────────────────────────────────────
    drift_report_dir: str = "monitoring/reports"
    drift_alert_threshold: float = 0.4
    shap_background_samples: int = 100
    fairness_sensitive_attributes: str = "driver_id"
    fairness_min_dp_diff: float = 0.1

    # ── Validators ────────────────────────────────────────
    @field_validator("secret_key")
    @classmethod
    def _validate_secret(cls, v: SecretStr) -> SecretStr:
        if v.get_secret_value() in {"", "changeme", "secret"}:
            raise ValueError("SECRET_KEY must not be a placeholder")
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
            errors.append("API_KEYS must be rotated in production")
        if not self.allowed_origins or "*" in self.allowed_origins:
            errors.append("ALLOWED_ORIGINS must be an explicit allow-list in production")
        if self.encryption_key.get_secret_value() == base64.urlsafe_b64encode(b"0" * 32).decode():
            errors.append("ENCRYPTION_KEY must be set in production")
        if self.secrets_backend == "env" and self.oidc_enabled is False:
            errors.append("Use SECRETS_BACKEND=vault or enable OIDC in production")
        if errors:
            raise ValueError("Production configuration invalid:\n  - " + "\n  - ".join(errors))
        return self

    # ── Helpers ───────────────────────────────────────────
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    def default_api_key(self) -> str:
        keys = self.api_key_set()
        return next(iter(keys)) if keys else ""

    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def pii_fields_list(self) -> list[str]:
        return [f.strip() for f in self.pii_fields.split(",") if f.strip()]

    def fairness_attributes_list(self) -> list[str]:
        return [a.strip() for a in self.fairness_sensitive_attributes.split(",") if a.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def reload_settings() -> Settings:
    os.environ.pop("_SETTINGS_LOADED", None)
    get_settings.cache_clear()
    return get_settings()
