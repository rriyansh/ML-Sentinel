"""
Locust load test.

Run:
    locust -f tests/load/locustfile.py --host http://localhost:8000
"""
from __future__ import annotations

import os

from locust import HttpUser, between, task

API_KEY = os.getenv("LOAD_TEST_API_KEY", "dev-key-abc123")


class MLSentinelUser(HttpUser):
    wait_time = between(0.1, 1.0)

    def on_start(self) -> None:
        r = self.client.post("/auth/token", json={"api_key": API_KEY})
        r.raise_for_status()
        self.headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    @task(10)
    def predict_single(self) -> None:
        self.client.post(
            "/api/v1/predict",
            json={"conv_rate": 0.5, "acc_rate": 0.9},
            headers=self.headers,
        )

    @task(3)
    def predict_batch(self) -> None:
        self.client.post(
            "/api/v1/predict/batch",
            json={"records": [
                {"conv_rate": 0.1, "acc_rate": 0.2},
                {"conv_rate": 0.3, "acc_rate": 0.4},
                {"conv_rate": 0.5, "acc_rate": 0.6},
            ]},
            headers=self.headers,
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/health/liveness")

    @task(1)
    def metrics(self) -> None:
        self.client.get("/metrics")
