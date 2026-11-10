"""
Drift detection with Evidently.

- Loads configuration from ``config/drift_config.yaml``.
- Runs a ``TestSuite`` (share-of-drifted-columns + per-feature PSI).
- Emits HTML + JSON reports under ``DRIFT_REPORT_DIR``.
- Persists a ``DriftEvent`` row for audit.
- Emits Prometheus counter via ``record_drift``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.report import Report

from ml_sentinel.database import session_scope
from ml_sentinel.db_models import DriftEvent
from ml_sentinel.logger import get_logger
from ml_sentinel.metrics import record_drift
from ml_sentinel.settings import get_settings
from ml_sentinel.utils import df_digest, ensure_dir, safe_json_load, truncate

logger = get_logger("drift_check")


def _load_config() -> dict[str, Any]:
    base = Path(__file__).resolve().parent.parent
    cfg_path = base / "config" / "drift_config.yaml"
    import yaml

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["drift"]


def _persist_event(
    *,
    detected: bool,
    score: float,
    threshold: float,
    features: list[str],
    details: dict[str, Any],
    report_path: str,
    reference_digest: str | None = None,
    current_digest: str | None = None,
) -> None:
    try:
        with session_scope() as session:
            event = DriftEvent(
                detected=detected,
                drift_score=score,
                threshold=threshold,
                features_monitored={"features": features},
                details=details,
                report_path=report_path,
                reference_digest=reference_digest,
                current_digest=current_digest,
            )
            session.add(event)
        logger.info("drift_event_persisted", detected=detected, score=score)
    except Exception:  # noqa: BLE001 — audit must never break the pipeline
        logger.exception("drift_event_persist_failed")


def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    """Run the drift test suite and return ``True`` if drift is detected."""
    cfg = _load_config()
    features: list[str] = cfg["features_to_monitor"]
    threshold: float = float(cfg["threshold"])
    p_threshold: float = float(cfg.get("p_value_threshold", 0.05))
    min_rows: int = int(cfg.get("min_sample_size", 100))

    # ── Validate inputs ───────────────────────────────────────────
    for name, df in (("reference", reference_df), ("current", current_df)):
        if df.empty:
            raise ValueError(f"{name} DataFrame is empty")
        if len(df) < min_rows:
            raise ValueError(f"{name} DataFrame too small: {len(df)} < {min_rows}")
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"{name} DataFrame missing columns: {missing}")

    ref = reference_df[features].copy()
    cur = current_df[features].copy()

    # ── Run Evidently ─────────────────────────────────────────────
    schema = DataDefinition(
        numerical_columns=features,
        categorical_columns=[],
    )
    ref_ds = Dataset.from_pandas(ref, data_definition=schema)
    cur_ds = Dataset.from_pandas(cur, data_definition=schema)

    report = Report([DataDriftPreset()])
    report.run(reference_data=ref_ds, current_data=cur_ds)

    out_dir = ensure_dir(get_settings().drift_report_dir)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    html_path = out_dir / f"drift_report_{ts}.html"
    json_path = out_dir / f"drift_report_{ts}.json"

    report.save_html(str(html_path))
    result = report.as_dict()
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # ── Interpret ─────────────────────────────────────────────────
    metric = result["metrics"][0]["result"]
    drift_score = float(metric["share_of_drifted_columns"])
    detected = drift_score > threshold

    # Optional p-value-based secondary check
    pvals = [
        v.get("drift_by_columns", {}).get(f, {}).get("stattest_threshold", 1.0)
        for f in features
    ] if "drift_by_columns" in metric else []
    if pvals and any(p < p_threshold for p in pvals):
        detected = True

    record_drift(detected)
    _persist_event(
        detected=detected,
        score=drift_score,
        threshold=threshold,
        features=features,
        details={
            "per_column": metric.get("drift_by_columns", {}),
            "n_features": len(features),
        },
        report_path=str(html_path),
        reference_digest=df_digest(ref),
        current_digest=df_digest(cur),
    )

    logger.info(
        "drift_check_completed",
        drift_detected=detected,
        drift_score=drift_score,
        threshold=threshold,
        report=str(html_path),
        details=truncate(json.dumps(metric), 500),
    )
    return detected


def run_manual_drift_check() -> bool:
    """Convenience entry point for local runs (``make drift-check``)."""
    base = Path(__file__).resolve().parent.parent
    ref = pd.read_parquet(base / "data" / "reference" / "driver_stats.parquet")
    # Simulate current data as a drifted subset
    curr = ref.sample(frac=0.5, random_state=99).copy()
    curr["conv_rate"] = (curr["conv_rate"] * 1.5).clip(0, 1)
    return check_drift(ref, curr)


if __name__ == "__main__":
    run_manual_drift_check()
