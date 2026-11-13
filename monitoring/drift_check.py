"""
Drift detection with Evidently (TestSuite) + audit + Prometheus counter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.report import Report

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import record_drift
from ml_sentinel.core.settings import get_settings
from ml_sentinel.core.utils import df_digest, ensure_dir, truncate
from ml_sentinel.domain.entities import DriftResult
from ml_sentinel.infrastructure.database import session_scope
from ml_sentinel.infrastructure.db_models import DriftEvent

logger = get_logger("drift_check")


def _load_config() -> dict[str, Any]:
    base = Path(__file__).resolve().parent.parent
    with (base / "config" / "drift_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["drift"]


def _persist(result: DriftResult, ref_digest: str, cur_digest: str) -> None:
    try:
        with session_scope() as session:
            session.add(DriftEvent(
                detected=result.detected,
                drift_score=result.drift_score,
                threshold=result.threshold,
                features_monitored={"features": result.features},
                details=result.details,
                report_path=result.report_path,
                reference_digest=ref_digest,
                current_digest=cur_digest,
            ))
    except Exception:  # noqa: BLE001
        logger.exception("drift_persist_failed")


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    cfg = _load_config()
    features: list[str] = cfg["features_to_monitor"]
    threshold: float = float(cfg["threshold"])
    p_threshold: float = float(cfg.get("p_value_threshold", 0.05))
    min_rows = int(cfg.get("min_sample_size", 100))

    for name, df in (("reference", reference_df), ("current", current_df)):
        if df.empty:
            raise ValueError(f"{name} is empty")
        if len(df) < min_rows:
            raise ValueError(f"{name} too small: {len(df)} < {min_rows}")
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    ref = reference_df[features].copy()
    cur = current_df[features].copy()

    schema = DataDefinition(numerical_columns=features, categorical_columns=[])
    report = Report([DataDriftPreset()])
    report.run(reference_data=Dataset.from_pandas(ref, data_definition=schema),
               current_data=Dataset.from_pandas(cur, data_definition=schema))

    out_dir = ensure_dir(get_settings().drift_report_dir)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    html_path = out_dir / f"drift_report_{ts}.html"
    json_path = out_dir / f"drift_report_{ts}.json"
    report.save_html(str(html_path))
    result_dict = report.as_dict()
    json_path.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")

    metric = result_dict["metrics"][0]["result"]
    drift_score = float(metric["share_of_drifted_columns"])
    detected = drift_score > threshold

    per_column = metric.get("drift_by_columns", {})
    pvals = [v.get("stattest_threshold", 1.0) for v in per_column.values()]
    if pvals and any(p < p_threshold for p in pvals):
        detected = True

    result = DriftResult(
        detected=detected,
        drift_score=drift_score,
        threshold=threshold,
        features=features,
        details={"per_column": per_column, "n_features": len(features)},
        report_path=str(html_path),
    )
    record_drift(detected)
    _persist(result, df_digest(ref), df_digest(cur))
    logger.info("drift_check_done", detected=detected, drift_score=drift_score,
                threshold=threshold, report=str(html_path),
                details=truncate(json.dumps(metric), 500))
    return result


def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    return detect_drift(reference_df, current_df).detected


def run_manual_drift_check() -> bool:
    base = Path(__file__).resolve().parent.parent
    ref = pd.read_parquet(base / "data" / "reference" / "driver_stats.parquet")
    curr = ref.sample(frac=0.5, random_state=99).copy()
    curr["conv_rate"] = (curr["conv_rate"] * 1.5).clip(0, 1)
    return check_drift(ref, curr)


if __name__ == "__main__":
    run_manual_drift_check()
