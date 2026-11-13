"""
Production Airflow DAG with TaskFlow API, drift branching,
quality gate (champion/challenger + fairness), and promotion.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import dag, task, task_group
from airflow.exceptions import AirflowFailException

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger("airflow")
settings = get_settings()


def _alert(context: dict, level: str = "ERROR") -> None:
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook:
        return
    try:
        import httpx

        ti = context.get("task_instance")
        msg = {
            "text": f":rotating_light: *[{level}]* `{context.get('dag').dag_id}.{ti.task_id if ti else '-'}`\n"
                    f"Run: {context.get('run_id')}\nLogs: {ti.log_url if ti else '-'}",
        }
        httpx.post(webhook, json=msg, timeout=5.0)
    except Exception:  # noqa: BLE001
        logger.exception("slack_alert_failed")


def _on_failure(context: dict) -> None:
    logger.error("task_failed", run_id=context.get("run_id"))
    _alert(context)


default_args = {
    "owner": "ml-sentinel",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
    "on_failure_callback": _on_failure,
    "sla": timedelta(hours=3),
}


@dag(
    dag_id="ml_sentinel_production_pipeline",
    description="Drift → Retrain → Gate (metrics+fairness) → Promote",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "drift", "retraining", "champion-challenger"],
)
def ml_sentinel_pipeline():

    @task(retries=2)
    def fetch_data() -> dict:
        ref = os.getenv("REFERENCE_DATA_PATH", "data/reference/driver_stats.parquet")
        cur = os.getenv("CURRENT_DATA_PATH", "data/current/latest.parquet")
        if not os.path.exists(ref):
            raise AirflowFailException(f"Missing reference: {ref}")
        if not os.path.exists(cur):
            df = pd.read_parquet(ref).sample(frac=0.5, random_state=99).copy()
            df["conv_rate"] = (df["conv_rate"] * 1.5).clip(0, 1)
            os.makedirs(os.path.dirname(cur), exist_ok=True)
            df.to_parquet(cur, index=False)
        return {"reference": ref, "current": cur}

    @task.branch
    def check_drift(paths: dict) -> str:
        from monitoring.drift_check import check_drift

        detected = check_drift(
            pd.read_parquet(paths["reference"]),
            pd.read_parquet(paths["current"]),
        )
        return "retrain_model" if detected else "skip_retrain"

    @task
    def skip_retrain() -> None:
        logger.info("no_drift_skip_retrain")

    @task(retries=1, execution_timeout=timedelta(hours=1))
    def retrain_model() -> dict:
        from mlflow.train import train_model

        return train_model()

    @task.short_circuit
    def quality_gate(challenger: dict) -> bool:
        import mlflow
        from mlflow.tracking import MlflowClient

        from ml_sentinel.domain.entities import ModelMetrics
        from ml_sentinel.domain.services import PromotionPolicy

        policy = json.loads(open("config/model_promotion_policy.json", encoding="utf-8").read())
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        champion_metrics = None
        try:
            prod = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
            if prod:
                run = client.get_run(prod[0].run_id)
                m = run.data.metrics
                champion_metrics = ModelMetrics(
                    f1=float(m.get("f1", 0.0)),
                    precision=float(m.get("precision", 0.0)),
                    recall=float(m.get("recall", 0.0)),
                )
        except Exception:  # noqa: BLE001
            logger.warning("champion_lookup_failed")

        challenger_m = ModelMetrics(
            f1=float(challenger["f1"]),
            precision=float(challenger["precision"]),
            recall=float(challenger["recall"]),
        )
        policy_obj = PromotionPolicy(policy)
        ok, reason = policy_obj.evaluate(challenger_m, champion_metrics)

        # Fairness gate
        if policy.get("fairness", {}).get("enabled"):
            from monitoring.fairness import run_fairness_check

            fr = run_fairness_check()
            if fr and not fr.passed:
                logger.warning("fairness_gate_failed", dp=fr.demographic_parity_diff)
                ok = False
                reason = f"fairness_failed:{reason}"

        logger.info("quality_gate", passed=ok, reason=reason,
                    champion=champion_metrics.as_dict() if champion_metrics else None,
                    challenger=challenger_m.as_dict())
        return ok

    @task
    def promote_model(challenger: dict) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        from ml_sentinel.infrastructure.database import session_scope
        from ml_sentinel.infrastructure.db_models import ModelPromotion

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        pending = client.get_latest_versions(settings.mlflow_model_name, stages=["None"])
        if not pending:
            raise AirflowFailException("No pending model")
        new_version = pending[0].version
        prev = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
        prev_version = prev[0].version if prev else None

        client.transition_model_version_stage(
            name=settings.mlflow_model_name, version=new_version,
            stage="Production", archive_existing_versions=True,
        )
        with session_scope() as session:
            session.add(ModelPromotion(
                model_name=settings.mlflow_model_name,
                new_version=new_version, previous_version=prev_version,
                approved=True, challenger_metrics=challenger,
                policy={"primary": "f1"}, reason="passed_quality_gate",
            ))
        logger.info("model_promoted", new_version=new_version, previous_version=prev_version)

    @task_group
    def retraining_branch():
        m = retrain_model()
        g = quality_gate(m)
        p = promote_model(m)
        g >> p

    paths = fetch_data()
    branch = check_drift(paths)
    skip = skip_retrain()

    branch >> [skip, retraining_branch()]


dag = ml_sentinel_pipeline()
