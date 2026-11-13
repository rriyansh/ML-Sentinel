"""Serving API tests."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEYS", "test-key-serving")
os.environ.setdefault("SECRET_KEY", "test-secret-key-with-min-32-characters-xxxx")
os.environ.setdefault("ENCRYPTION_KEY", "kNfK8XbRw3h9nPZQhG7jCkYe0gY2mAVa6IioK5zPq4U=")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_serving.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

from serving import app as app_module  # noqa: E402


class _Dummy:
    def predict(self, rows):
        return [float(sum(r)) for r in rows]


@pytest.fixture()
def client(monkeypatch):
    app_module._state.update(model=_Dummy(), name="MLSentinelModel", version="1", stage="Production")
    monkeypatch.setattr(app_module, "ping_db", lambda: True)
    return TestClient(app_module.app)


def test_liveness(client):
    r = client.get("/health/liveness")
    assert r.status_code == 200


def test_readiness(client):
    r = client.get("/health/readiness")
    assert r.status_code == 200


def test_token_and_predict(client):
    r = client.post("/auth/token", json={"api_key": "test-key-serving"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    r = client.post("/api/v1/predict",
                    json={"conv_rate": 0.5, "acc_rate": 0.9},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_predict_requires_auth(client):
    r = client.post("/api/v1/predict", json={"conv_rate": 0.5, "acc_rate": 0.9})
    assert r.status_code == 401


def test_invalid_input(client):
    r = client.post("/api/v1/predict",
                    json={"conv_rate": 5.0, "acc_rate": 0.9},
                    headers={"X-API-Key": "test-key-serving"})
    assert r.status_code == 422


def test_security_headers(client):
    r = client.get("/health/liveness")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ml_sentinel" in r.text
