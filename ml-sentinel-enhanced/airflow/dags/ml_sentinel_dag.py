import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.branch import BaseBranchOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mlflow.train import train
from monitoring.drift_check import check_drift

default_args = {
    "owner": "ml-sentinel-enhanced",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_sentinel_pipeline_enhanced",
    default_args=default_args,
    description="ML-Sentinel Enhanced: training, quality gate, drift monitoring with branching",
    schedule=timedelta(hours=24),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml-sentinel-enhanced"],
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train,
        op_kwargs={
            "data_path": "ml-sentinel-enhanced/feast_repo/data/driver_stats.parquet"
        },
    )

    detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=check_drift,
    )

    class QualityGate(BaseBranchOperator):
        def choose_branch(self, context):
            ti = context["ti"]
            drift_result = ti.xcom_pull(task_ids="detect_drift")
            if drift_result and drift_result.get("dataset_drift_detected"):
                return "retrain_model"
            return "skip_retrain"

    quality_gate = QualityGate(task_id="quality_gate")

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=train,
        trigger_rule=TriggerRule.NONE_SKIPPED,
    )

    skip_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=lambda: print("No drift detected. Skipping retrain."),
        trigger_rule=TriggerRule.NONE_SKIPPED,
    )

    train_model >> detect_drift >> quality_gate >> [retrain_model, skip_retrain]
