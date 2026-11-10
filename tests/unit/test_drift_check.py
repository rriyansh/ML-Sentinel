"""Drift detection tests (Evidently mocked lightly via real computation)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_drift.db")
os.environ.setdefault("DRIFT_REPORT_DIR", "/tmp/drift_reports")

from monitoring import drift_check  # noqa: E402


def test_no_drift_returns_false(reference_df):
    detected = drift_check.check_drift(reference_df, reference_df)
    assert detected is False


def test_significant_drift_returns_true(reference_df, drifted_df):
    detected = drift_check.check_drift(reference_df, drifted_df)
    assert detected is True


def test_empty_data_raises(reference_df):
    with pytest.raises(ValueError, match="empty"):
        drift_check.check_drift(reference_df, pd.DataFrame())


def test_too_small_data_raises(reference_df):
    small = reference_df.head(5)
    with pytest.raises(ValueError, match="too small"):
        drift_check.check_drift(reference_df, small)
