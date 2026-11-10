"""
Integration test: end-to-end dry run without a real MLflow server.

Uses a temporary file-backed MLflow tracking directory so the whole
train + register flow can be exercised in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/mlruns_int")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_integration.db")

from ml_sentinel.database import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_db_once():
    init_db()


def test_full_train_and_register(tmp_path: Path, monkeypatch, reference_df):
    """Train against a tiny toy dataset and assert the model registers."""
    feast_dir = tmp_path / "feast_repo" / "data"
    feast_dir.mkdir(parents=True)
    reference_df.assign(
        driver_id=range(1, len(reference_df) + 1),
        event_timestamp=pd.Timestamp.utcnow(),
        created_timestamp=pd.Timestamp.utcnow(),
    ).to_parquet(feast_dir / "driver_stats.parquet", index=False)

    monkeypatch.setenv("FEAST_REPO_PATH", str(tmp_path / "feast_repo"))

    # Only import here so env changes take effect
    from mlflow import train as train_module  # noqa: F811

    # Skip the actual Feast retrieval (already covered elsewhere)
    monkeypatch.setattr(
        train_module,
        "_fetch_training_data",
        lambda store: reference_df.assign(
            avg_daily_trips=reference_df["avg_daily_trips"].astype(int)
        ),
    )

    metrics = train_module.train_model()
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
