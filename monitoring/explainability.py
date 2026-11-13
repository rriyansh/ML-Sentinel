"""
SHAP-based explainability for model predictions.

Produces per-request explanations and batch global importance.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import shap

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import EXPLANATIONS
from ml_sentinel.core.settings import get_settings

logger = get_logger("explainability")


def explain_prediction(
    model: object, background: pd.DataFrame, instance: pd.DataFrame, top_k: int = 3
) -> dict:
    """
    Compute SHAP values for a single instance using KernelExplainer.
    """
    try:
        explainer = shap.KernelExplainer(
            lambda x: model.predict(pd.DataFrame(x, columns=instance.columns)),
            background.sample(min(50, len(background)), random_state=42),
        )
        shap_values = explainer.shap_values(instance, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        values = np.asarray(shap_values).flatten()

        contributions = {
            col: float(v) for col, v in zip(instance.columns, values, strict=False)
        }
        sorted_contrib = dict(
            sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        )
        EXPLANATIONS.labels(status="success").inc()
        return {
            "contributions": sorted_contrib,
            "base_value": float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                                else explainer.expected_value[0]),
        }
    except Exception as exc:  # noqa: BLE001
        EXPLANATIONS.labels(status="error").inc()
        logger.exception("shap_failed", error=str(exc))
        raise


def run_shap_explanations() -> None:
    """Batch-mode utility: generate a global SHAP summary for the current model."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        model = mlflow.pyfunc.load_model(f"models:/{settings.mlflow_model_name}/Production")
    except Exception:  # noqa: BLE001
        logger.warning("no_production_model_skipping_shap")
        return

    base = Path(__file__).resolve().parent.parent
    ref = pd.read_parquet(base / "data" / "reference" / "driver_stats.parquet")
    X = ref[["conv_rate", "acc_rate"]].sample(min(100, len(ref)), random_state=42)

    explainer = shap.KernelExplainer(
        lambda x: model.predict(pd.DataFrame(x, columns=X.columns)),
        X.head(20),
    )
    shap_vals = explainer.shap_values(X, nsamples=100)
    importance = dict(zip(X.columns, np.abs(shap_vals).mean(axis=0), strict=False))

    out = Path(settings.drift_report_dir) / "shap_global.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: float(v) for k, v in importance.items()}, indent=2))
    logger.info("shap_global_saved", path=str(out), importance=importance)


if __name__ == "__main__":
    run_shap_explanations()
