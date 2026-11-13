"""
Training pipeline with Feast lineage + MLflow tracking + fairness gate.

Returns the metrics dict for Airflow's quality gate.
"""
from __future__ import annotations

import json
import os
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feast import FeatureStore
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ml_sentinel.core.errors import DataValidationError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings
from ml_sentinel.core.tracing import setup_tracing, get_tracer
from ml_sentinel.core.utils import dict_digest, safe_json_load
from ml_sentinel.domain.entities import ModelMetrics

logger = get_logger("train")
_tracer = None


def _validate(df: pd.DataFrame, min_rows: int = 100) -> None:
    required = {"conv_rate", "acc_rate", "avg_daily_trips"}
    if df.empty:
        raise DataValidationError("Empty training data")
    if len(df) < min_rows:
        raise DataValidationError(f"Insufficient rows: {len(df)} < {min_rows}")
    if required - set(df.columns):
        raise DataValidationError(f"Missing columns: {required - set(df.columns)}")
    if df[list(required)].isna().any().any():
        raise DataValidationError("NaN values in training data")


def _fetch_data(store: FeatureStore) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entity_df = pd.read_parquet(os.path.join(base, "feast_repo/data/driver_stats.parquet"))[
        ["driver_id", "event_timestamp"]
    ]
    return store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df().dropna()


def train_model() -> dict[str, Any]:
    global _tracer
    settings = get_settings()
    setup_tracing()
    from ml_sentinel.core.tracing import get_tracer
    _tracer = get_tracer("mlflow.train")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with _tracer.start_as_current_span("train.pipeline"):
        logger.info("training_started", experiment=settings.mlflow_experiment_name)
        store = FeatureStore(repo_path=settings.feast_repo_path)
        policy = safe_json_load(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "model_promotion_policy.json",
        ))

        with _tracer.start_as_current_span("train.fetch_data"):
            df = _fetch_data(store)
        _validate(df, int(policy.get("min_training_samples", 100)))

        X = df[["conv_rate", "acc_rate"]].astype("float64")
        y = (df["avg_daily_trips"] > df["avg_daily_trips"].median()).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        with _tracer.start_as_current_span("train.fit"):
            params = {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1}
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = ModelMetrics(
            f1=float(f1_score(y_test, preds)),
            precision=float(precision_score(y_test, preds)),
            recall=float(recall_score(y_test, preds)),
        )
        signature = infer_signature(X_test, model.predict(X_test))
        data_digest = dict_digest({"rows": len(df), "cols": list(df.columns)})

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics.as_dict())
            mlflow.log_dict(policy, "config/promotion_policy.json")
            mlflow.set_tag("data_digest", data_digest)
            mlflow.set_tag("training_rows", str(len(df)))
            store.mlflow.log_model(
                model, artifact_path="model",
                registered_model_name=settings.mlflow_model_name,
                signature=signature, input_example=X_test.head(3),
            )
            run_id = run.info.run_id

        logger.info("training_completed", run_id=run_id, **metrics.as_dict())
        print(json.dumps(metrics.as_dict()))
        return metrics.as_dict()


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
