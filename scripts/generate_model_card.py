"""
Auto-generate a Model Card (Markdown) from MLflow metadata + fairness report.

Follows the Model Cards for Model Reporting format (Mitchell et al., 2019).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger("model_card")


def main() -> None:
    settings = get_settings()
    base = Path(__file__).resolve().parent.parent

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        versions = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
        if not versions:
            logger.warning("no_production_model")
            version = "N/A"
            metrics = {}
            run_id = "N/A"
        else:
            v = versions[0]
            version = v.version
            run_id = v.run_id
            run = client.get_run(run_id)
            metrics = run.data.metrics
    except Exception:  # noqa: BLE001
        logger.exception("model_card_mlflow_failed")
        version, metrics, run_id = "N/A", {}, "N/A"

    fairness = {}
    fp = base / settings.drift_report_dir / "fairness_report.json"
    if fp.exists():
        fairness = json.loads(fp.read_text())

    card = f"""# Model Card — {settings.mlflow_model_name}

> Auto-generated on {datetime.now(timezone.utc).isoformat()} — Run ID: `{run_id}`

## Model Details

| Field | Value |
|---|---|
| Name | `{settings.mlflow_model_name}` |
| Version | `{version}` |
| Stage | Production |
| Framework | scikit-learn (RandomForestClassifier) |
| Environment | {settings.environment} |

## Intended Use

Predicts daily trip volume category from driver hourly statistics.
**Not** intended for autonomous dispatch decisions.

## Metrics (Production)

| Metric | Value |
|---|---|
{chr(10).join(f"| {k} | {v:.4f} |" for k, v in metrics.items()) or "| — | — |"}

## Fairness

| Attribute | Demographic Parity Diff | Equalized Odds Diff | Passed |
|---|---|---|---|
| {fairness.get('attribute', 'N/A')} | {fairness.get('dp_diff', 'N/A')} | {fairness.get('eo_diff', 'N/A')} | {fairness.get('passed', 'N/A')} |

## Ethical Considerations

- Model is retrained automatically when Evidently detects drift.
- Champion/Challenger gate prevents silent regressions.
- Fairness gate rejects promotion when demographic parity diff exceeds threshold.

## Caveats & Limitations

- Trained on synthetic reference data for demo purposes.
- Performance may degrade on unseen distributions — monitor drift continuously.

## Contact

- Owner: ML-Sentinel Team
- Contact: platform@ml-sentinel.dev
"""

    out = base / "MODEL_CARD.md"
    out.write_text(card, encoding="utf-8")
    logger.info("model_card_generated", path=str(out))


if __name__ == "__main__":
    main()
