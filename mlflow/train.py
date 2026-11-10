import mlflow
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from src.ml_sentinel.logger import setup_logger

logger = setup_logger("train")

def train_model():
    logger.info("Starting training pipeline...")
    store = FeatureStore(repo_path="feast_repo/")

    entity_df = pd.read_parquet("feast_repo/data/driver_stats.parquet")
    entity_df = entity_df[["driver_id", "event_timestamp"]]

    # Feast will auto-log this retrieval to the active MLflow run
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df()

    X = training_df[["conv_rate", "acc_rate"]]
    y = training_df["avg_daily_trips"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (using simulated classification metrics for the gate)
    preds = model.predict(X_test)
    f1 = f1_score((y_test > y_test.mean()).astype(int), (preds > y_test.mean()).astype(int))
    precision = precision_score((y_test > y_test.mean()).astype(int), (preds > y_test.mean()).astype(int))
    recall = recall_score((y_test > y_test.mean()).astype(int), (preds > y_test.mean()).astype(int))

    mlflow.set_experiment("ml_sentinel_production")
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # This registers the model AND propagates feature lineage
        store.mlflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="MLSentinelModel",
        )
        logger.info(f"Model registered. Run ID: {run.info.run_id}")

    return {"f1": f1, "precision": precision, "recall": recall}
