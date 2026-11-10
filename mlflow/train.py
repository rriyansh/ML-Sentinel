"""
Model training pipeline with Feast historical retrieval and MLflow tracking.

Key features:
- Fetches training data via Feast (auto-logged lineage to MLflow).
- Validates schema and minimum sample count before training.
- Logs model signature + input example for downstream serving safety.
- Registers the model in the MLflow Model Registry.
- Returns the metrics dict so Airflow's quality gate can consume it.
"""
from __future__ import annotations

import json
import os
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ml_sentinel.exceptions import DataValidationError
from ml_sentinel.logger import get_logger
from ml_sentinel.settings import get_settings
from ml_sentinel.utils import dict_digest, safe_json_load

logger = get_logger("train")

REQUIRED_COLUMNS = {"conv_rate", "acc_rate", "avg_daily_trips"}


def _validate_training_data(df: pd.DataFrame, min_rows: int = 100) -> None:
    if df.empty:
        raise DataValidationError("Training DataFrame is empty")
    if len(df) < min_rows:
        raise DataValidationError(
            f"Insufficient training data: {len(df)} < {min_rows} rows"
        )
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise DataValidationError("Training data contains NaN values")
    if not df["avg_daily_trips"].between(0, 10_000).all():
        raise DataValidationError("avg_daily_trips out of expected bounds")


def _load_promotion_policy() -> dict[str, Any]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return safe_json_load(os.path.join(base, "config", "model_promotion_policy.json"))


def _fetch_training_data(store: FeatureStore) -> pd.DataFrame:
    """Retrieve historical features from Feast (lineage auto-logged)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entity_path = os.path.join(base, "feast_repo", "data", "driver_stats.parquet")

    entity_df = pd.read_parquet(entity_path)[["driver_id", "event_timestamp"]]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df()

    return training_df.dropna(subset=list(REQUIRED_COLUMNS))


def train_model() -> dict[str, Any]:
    """Train, register, and return the challenger metrics."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("training_started", experiment=settings.mlflow_experiment_name)

    store = FeatureStore(repo_path=settings.feast_repo_path)
    policy = _load_promotion_policy()

    training_df = _fetch_training_data(store)
    _validate_training_data(training_df, min_rows=int(policy.get("min_training_samples", 100)))

    X = training_df[["conv_rate", "acc_rate"]].astype("float64")
    # Simulated label so we can use classification metrics for the demo gate
    y = (training_df["avg_daily_trips"] > training_df["avg_daily_trips"].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
    }
    data_digest = dict_digest({"rows": len(training_df), "columns": list(training_df.columns)})

    signature = infer_signature(X_test, model.predict(X_test))

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(policy, "config/promotion_policy.json")
        mlflow.set_tag("data_digest", data_digest)
        mlflow.set_tag("training_rows", str(len(training_df)))

        store.mlflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
            signature=signature,
            input_example=X_test.head(3),
        )
        run_id = run.info.run_id

    logger.info(
        "training_completed",
        run_id=run_id,
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        rows=len(training_df),
    )
    # Airflow reads this JSON from XCom
    print(json.dumps(metrics))
    return metrics


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
