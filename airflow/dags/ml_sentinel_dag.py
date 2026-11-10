"""
Production Airflow DAG for ML-Sentinel.

Workflow:
  1. fetch_data        — pull reference + current windows
  2. check_drift       — run Evidently (branches based on outcome)
  3. retrain_model     — feast + MLflow training
  4. quality_gate      — champion/challenger policy check
  5. promote_model     — MLflow registry stage transition

Reliability:
  - Idempotent tasks
  - SLA + callbacks (on_failure, on_sla_miss)
  - Retries with exponential backoff
  - Structured logs via structlog
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import dag, task, task_group
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

from ml_sentinel.logger import get_logger
from ml_sentinel.settings import get_settings

logger = get_logger("airflow")
settings = get_settings()

# ── Callbacks ───────────────────────────────────────────────────────
def _alert(context: dict, level: str = "ERROR") -> None:
    """Best-effort Slack alert; no-op if webhook is missing."""
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook:
        return
    try:
        import httpx

        dag_id = context.get("dag").dag_id if context.get("dag") else "unknown"
        task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown"
        msg = {
            "text": f":rotating_light: *[{level}]* `{dag_id}.{task_id}`\n"
                    f"Run: {context.get('run_id')}\n"
                    f"Logs: {context.get('task_instance').log_url if context.get('task_instance') else '-'}"
        }
        httpx.post(webhook, json=msg, timeout=5.0)
    except Exception:  # noqa: BLE001
        logger.exception("slack_alert_failed")


def _on_failure(context: dict) -> None:
    logger.error("task_failed", **{k: str(v)[:200] for k, v in context.items() if k in {"task_instance", "exception", "run_id"}})
    _alert(context, level="ERROR")


def _on_sla_miss(dag, task_list, blocking_task_list, slas, blocking_tis):  # noqa: ANN001
    logger.warning("sla_miss", dag=dag.dag_id, tasks=task_list)


def _on_success(context: dict) -> None:
    logger.info("dag_succeeded", dag=context.get("dag").dag_id)


# ── DAG ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "ml-sentinel",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
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
    description="Drift → Retrain → Gate → Promote",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    sla_miss_callback=_on_sla_miss,
    on_success_callback=_on_success,
    tags=["mlops", "drift", "retraining", "champion-challenger"],
    doc_md=__doc__,
)
def ml_sentinel_pipeline():

    @task(task_id="fetch_data", retries=2)
    def fetch_data() -> dict:
        """Return paths to the reference & current dataset files."""
        ref_path = os.getenv(
            "REFERENCE_DATA_PATH", "data/reference/driver_stats.parquet"
        )
        cur_path = os.getenv(
            "CURRENT_DATA_PATH", "data/current/latest.parquet"
        )
        if not os.path.exists(ref_path):
            raise AirflowFailException(f"Reference data missing: {ref_path}")

        # If no live data is present, fall back to a drifted sample of reference
        if not os.path.exists(cur_path):
            logger.warning("current_data_missing_using_synthetic")
            df = pd.read_parquet(ref_path).sample(frac=0.5, random_state=99).copy()
            df["conv_rate"] = (df["conv_rate"] * 1.5).clip(0, 1)
            os.makedirs(os.path.dirname(cur_path), exist_ok=True)
            df.to_parquet(cur_path, index=False)

        return {"reference": ref_path, "current": cur_path}

    @task.branch(task_id="check_drift")
    def check_drift_task(paths: dict) -> str:
        """Run drift detection; return the next task id to execute."""
        from monitoring.drift_check import check_drift

        ref = pd.read_parquet(paths["reference"])
        cur = pd.read_parquet(paths["current"])
        detected = check_drift(ref, cur)
        logger.info("branch_decision", drift_detected=detected)
        return "retrain_model" if detected else "skip_retrain"

    @task(task_id="skip_retrain")
    def skip_retrain() -> None:
        logger.info("no_drift_skipping_retraining")

    @task(
        task_id="retrain_model",
        retries=1,
        retry_delay=timedelta(minutes=10),
        execution_timeout=timedelta(hours=1),
    )
    def retrain_model() -> dict:
        from mlflow.train import train_model

        metrics = train_model()
        logger.info("challenger_trained", **metrics)
        return metrics

    @task.short_circuit(task_id="quality_gate")
    def quality_gate(challenger: dict) -> bool:
        """Return False (skip downstream) if challenger loses to champion."""
        import mlflow
        from mlflow.tracking import MlflowClient

        policy = json.loads(
            open("config/model_promotion_policy.json", encoding="utf-8").read()
        )
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        champion: dict[str, float] = {}
        try:
            prod_versions = client.get_latest_versions(
                settings.mlflow_model_name, stages=["Production"]
            )
            if prod_versions:
                run = client.get_run(prod_versions[0].run_id)
                champion = {
                    k: float(v)
                    for k, v in run.data.metrics.items()
                    if k in {"f1", "precision", "recall"}
                }
        except Exception:  # noqa: BLE001
            logger.warning("champion_lookup_failed_treating_as_empty")

        if not champion:
            logger.info("no_champion_promoting_challenger")
            return True

        primary = policy.get("primary_metric", "f1")
        min_improvement = float(policy.get("min_primary_metric_improvement", 0.0))
        max_drops = policy.get("max_secondary_metric_drops", {})

        improved = (challenger[primary] - champion[primary]) >= min_improvement
        secondary_ok = all(
            (champion[m] - challenger[m]) <= float(tol)
            for m, tol in max_drops.items()
            if m in champion and m in challenger
        )

        passed = improved and secondary_ok
        logger.info(
            "quality_gate_result",
            passed=passed,
            champion=champion,
            challenger=challenger,
            primary_metric=primary,
        )
        return passed

    @task(task_id="promote_model")
    def promote_model(challenger: dict) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        from ml_sentinel.database import session_scope
        from ml_sentinel.db_models import ModelPromotion

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        pending = client.get_latest_versions(settings.mlflow_model_name, stages=["None"])
        if not pending:
            raise AirflowFailException("No pending model version to promote")
        new_version = pending[0].version

        prev_versions = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
        prev_version = prev_versions[0].version if prev_versions else None

        client.transition_model_version_stage(
            name=settings.mlflow_model_name,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )

        with session_scope() as session:
            session.add(
                ModelPromotion(
                    model_name=settings.mlflow_model_name,
                    new_version=new_version,
                    previous_version=prev_version,
                    approved=True,
                    challenger_metrics=challenger,
                    policy={"primary": "f1"},
                    reason="passed_quality_gate",
                )
            )

        logger.info(
            "model_promoted",
            new_version=new_version,
            previous_version=prev_version,
        )

    @task_group(group_id="retraining_branch")
    def retraining_branch():
        metrics = retrain_model()
        gate = quality_gate(metrics)
        promote = promote_model(metrics)
        gate >> promote

    # ── Wire up ────────────────────────────────────────────────
    paths = fetch_data()
    branch = check_drift_task(paths)
    skip = skip_retrain()

    branch >> [skip, retraining_branch()]


dag = ml_sentinel_pipeline()
