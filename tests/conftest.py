"""Shared pytest fixtures."""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret-key-with-enough-entropy-123456")
os.environ.setdefault("API_KEYS", "test-key-1,test-key-2")
os.environ.setdefault("ENCRYPTION_KEY", "kNfK8XbRw3h9nPZQhG7jCkYe0gY2mAVa6IioK5zPq4U=")
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("DRIFT_REPORT_DIR", "/tmp/drift_reports")


@pytest.fixture(scope="session")
def reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "conv_rate": [0.1, 0.2, 0.3, 0.4, 0.5] * 40,
            "acc_rate": [0.8, 0.85, 0.9, 0.95, 0.99] * 40,
            "avg_daily_trips": [10, 20, 30, 40, 50] * 40,
        }
    )


@pytest.fixture()
def drifted_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    df = reference_df.copy()
    df["conv_rate"] = (df["conv_rate"] * 5).clip(0, 1)
    df["acc_rate"] = (df["acc_rate"] * 0.01).clip(0, 1)
    return df
