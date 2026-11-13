"""
Contract test — asserts the serving API response matches a frozen schema.

Uses a lightweight snapshot approach instead of Pact to stay dependency-free.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEYS", "test-key-contract")
os.environ.setdefault("SECRET_KEY", "x" * 40)
os.environ.setdefault("ENCRYPTION_KEY", "kNfK8XbRw3h9nPZQhG7jCkYe0gY2mAVa6IioK5zPq4U=")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_contract.db")

from serving import app as app_module  # noqa: E402


class _Dummy:
    def predict(self, rows):
        return [float(sum(r)) for r in rows]


@pytest.fixture()
def client(monkeypatch):
    app_module._state.update(model=_Dummy(), name="MLSentinelModel", version="1", stage="Production")
    monkeypatch.setattr(app_module, "ping_db", lambda: True)
    return TestClient(app_module.app)


@pytest.mark.contract
def test_predict_contract(client):
    r = client.post("/api/v1/predict",
                    json={"conv_rate": 0.5, "acc_rate": 0.9},
                    headers={"X-API-Key": "test-key-contract"})
    assert r.status_code == 200
    body = r.json()
    required = {"predictions", "model_name", "model_version", "request_id", "latency_ms"}
    assert required.issubset(body.keys())
    assert isinstance(body["predictions"], list)
    assert isinstance(body["latency_ms"], (int, float))
