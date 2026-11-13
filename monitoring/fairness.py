"""
Fairness auditing: demographic parity + equalized odds.

Compares positive-prediction rates and TPR across groups defined by a
sensitive attribute.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import FAIRNESS_RUNS
from ml_sentinel.core.settings import get_settings
from ml_sentinel.domain.entities import FairnessResult
from ml_sentinel.infrastructure.database import session_scope
from ml_sentinel.infrastructure.db_models import FairnessAudit

logger = get_logger("fairness")


def _demographic_parity_diff(y_pred: np.ndarray, groups: pd.Series) -> float:
    rates = pd.Series(y_pred).groupby(groups.values).mean()
    return float(rates.max() - rates.min())


def _equalized_odds_diff(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups.values})
    tpr = df.groupby("g").apply(lambda g: g.loc[g["y"] == 1, "p"].mean() if (g["y"] == 1).any() else np.nan)
    return float(tpr.max() - tpr.min()) if not tpr.isna().all() else 0.0


def audit(
    y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, attribute: str
) -> FairnessResult:
    dp = _demographic_parity_diff(y_pred, groups)
    eo = _equalized_odds_diff(y_true, y_pred, groups)
    threshold = get_settings().fairness_min_dp_diff
    passed = dp <= threshold
    FAIRNESS_RUNS.labels(passed=str(passed).lower()).inc()
    return FairnessResult(
        attribute=attribute,
        demographic_parity_diff=dp,
        equalized_odds_diff=eo,
        passed=passed,
        details={"threshold": threshold},
    )


def run_fairness_check() -> FairnessResult | None:
    """Convenience batch entry point against the reference dataset."""
    settings = get_settings()
    base = Path(__file__).resolve().parent.parent
    ref = pd.read_parquet(base / "data" / "reference" / "driver_stats.parquet")
    if ref.empty:
        logger.warning("no_reference_data_for_fairness")
        return None

    y_true = (ref["avg_daily_trips"] > ref["avg_daily_trips"].median()).astype(int).values
    y_pred = (ref["conv_rate"] > 0.5).astype(int).values
    attribute = settings.fairness_attributes_list()[0] if settings.fairness_attributes_list() else "driver_id"
    groups = (ref[attribute] % 3).astype(str)  # toy group split for demo

    result = audit(y_true, y_pred, groups, attribute)

    out = Path(settings.drift_report_dir) / "fairness_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "attribute": result.attribute,
        "dp_diff": result.demographic_parity_diff,
        "eo_diff": result.equalized_odds_diff,
        "passed": result.passed,
    }, indent=2))

    try:
        with session_scope() as session:
            session.add(FairnessAudit(
                attribute=result.attribute,
                demographic_parity_diff=result.demographic_parity_diff,
                equalized_odds_diff=result.equalized_odds_diff,
                passed=result.passed,
                details=result.details,
            ))
    except Exception:  # noqa: BLE001
        logger.exception("fairness_persist_failed")

    logger.info("fairness_completed", **{
        "attribute": result.attribute,
        "dp": result.demographic_parity_diff,
        "eo": result.equalized_odds_diff,
        "passed": result.passed,
    })
    return result


if __name__ == "__main__":
    run_fairness_check()
