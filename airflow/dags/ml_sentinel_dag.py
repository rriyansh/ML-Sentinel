from datetime import datetime, timedelta
import json
import pandas as pd
from airflow.decorators import task, branch_task, dag
from airflow.models import Variable
from src.ml_sentinel.config import config
from monitoring.drift_check import check_drift
from mlflow.train import train_model

default_args = {
    "owner": "ml-sentinel",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="ml_sentinel_production_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "drift-detection", "champion-challenger"],
)
def ml_sentinel_pipeline():

    @task
    def fetch_data():
        """Fetches reference and current data."""
        reference_df = pd.read_parquet("data/reference/driver_stats.parquet")
        # In production, fetch current data from a live table
        current_df = pd.read_parquet("data/current/latest.parquet")
        return reference_df, current_df

    @branch_task
    def check_drift_branch(reference_df: pd.DataFrame, current_df: pd.DataFrame):
        """Returns the task_id to follow based on drift detection."""
        drift_detected = check_drift(reference_df, current_df)
        return "retrain_model" if drift_detected else "skip_retrain"

    @task
    def skip_retrain():
        print("No drift detected. Skipping retraining.")

    @task
    def retrain_model():
        """Retrains and returns the new model's metrics."""
        new_metrics = train_model()
        return new_metrics

    @task
    def quality_gate(new_metrics: dict):
        """Compares new model against the champion."""
        policy = config.promotion
        # In production, fetch champion metrics from MLflow Model Registry
        champion_metrics = {"f1": 0.75, "precision": 0.80, "recall": 0.70}

        f1_improved = (new_metrics["f1"] - champion_metrics["f1"]) >= policy["min_f1_improvement"]
        precision_ok = (champion_metrics["precision"] - new_metrics["precision"]) <= policy["max_precision_drop"]
        recall_ok = (champion_metrics["recall"] - new_metrics["recall"]) >= -policy["max_recall_drop"]

        if not (f1_improved and precision_ok and recall_ok):
            raise ValueError("Model failed the quality gate. Not promoting.")
        print(f"Quality gate passed. New F1: {new_metrics['f1']}")

    @task
    def promote_model():
        """Promotes the latest model to Production."""
        import mlflow
        client = mlflow.tracking.MlflowClient()
        # Get the latest version of the model
        latest_version = client.get_latest_versions("MLSentinelModel", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="MLSentinelModel",
            version=latest_version,
            stage="Production",
        )
        print(f"Promoted model version {latest_version} to Production.")

    # Define the workflow
    ref_df, curr_df = fetch_data()
    branch = check_drift_branch(ref_df, curr_df)
    skip = skip_retrain()
    retrain = retrain_model()
    gate = quality_gate(retrain)
    promote = promote_model()

    branch >> [skip, retrain]
    retrain >> gate >> promote

ml_sentinel_pipeline()
