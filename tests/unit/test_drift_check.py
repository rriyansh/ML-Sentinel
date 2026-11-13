"""Drift tests."""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_drift.db")
os.environ.setdefault("DRIFT_REPORT_DIR", "/tmp/drift_reports")
os.environ.setdefault("SECRET_KEY", "x" * 40)
os.environ.setdefault("ENCRYPTION_KEY", "kNfK8XbRw3h9nPZQhG7jCkYe0gY2mAVa6IioK5zPq4U=")

from monitoring import drift_check  # noqa: E402


def test_no_drift(reference_df):
    assert drift_check.check_drift(reference_df, reference_df) is False


def test_drift(reference_df, drifted_df):
    assert drift_check.check_drift(reference_df, drifted_df) is True


def test_empty_raises(reference_df):
    with pytest.raises(ValueError, match="empty"):
        drift_check.check_drift(reference_df, pd.DataFrame())


def test_small_raises(reference_df):
    with pytest.raises(ValueError, match="too small"):
        drift_check.check_drift(reference_df, reference_df.head(5))
