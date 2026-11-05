import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "ml_sentinel_driver_conversion"

FEATURES = ["conv_rate", "acc_rate", "avg_daily_trips"]
TARGET = "converted"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def train(data_path: str = "feast_repo/data/driver_stats.parquet") -> str:
    df = load_data(data_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="driver_conversion_rf") as run:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"MLflow run {run_id} - accuracy: {acc:.4f}")

    return run_id


if __name__ == "__main__":
    train()
