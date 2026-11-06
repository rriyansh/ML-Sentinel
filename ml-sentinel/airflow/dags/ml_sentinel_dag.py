from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mlflow.train import train
from monitoring.drift_check import check_drift

default_args = {
    "owner": "ml-sentinel",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_sentinel_pipeline",
    default_args=default_args,
    description="Production ML-Sentinel pipeline: training and drift monitoring",
    schedule=timedelta(hours=24),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml-sentinel"],
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train,
        op_kwargs={
            "data_path": "ml-sentinel/feast_repo/data/driver_stats.parquet"
        },
    )

    detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=check_drift,
    )

    train_model >> detect_drift
