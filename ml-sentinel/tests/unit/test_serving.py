from fastapi.testclient import TestClient

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from serving.app import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_missing_api_key():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"conv_rate": 0.5, "acc_rate": 0.9, "avg_daily_trips": 20},
    )
    assert response.status_code == 403
