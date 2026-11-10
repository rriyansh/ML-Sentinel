"""Tests for the configuration layer."""

from __future__ import annotations

import os

import pytest

from ml_sentinel.settings import Settings, reload_settings


def test_defaults_load():
    s = Settings()
    assert s.app_name == "ML-Sentinel"
    assert s.jwt_algorithm == "HS256"


def test_api_key_set_parses_csv():
    os.environ["API_KEYS"] = "a,b,c"
    try:
        s = Settings()
        assert s.api_key_set() == {"a", "b", "c"}
    finally:
        os.environ.pop("API_KEYS", None)


def test_production_rejects_debug(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SECRET_KEY", "x" * 40)
    monkeypatch.setenv("API_KEYS", "rotated-key-123")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
    with pytest.raises(ValueError, match="DEBUG must be false"):
        Settings()


def test_production_rejects_default_api_key(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("SECRET_KEY", "x" * 40)
    monkeypatch.setenv("API_KEYS", "dev-key-abc123")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
    with pytest.raises(ValueError, match="API_KEYS must be rotated"):
        Settings()


def test_reload_settings_clears_cache():
    a = reload_settings()
    b = reload_settings()
    assert a is not b
