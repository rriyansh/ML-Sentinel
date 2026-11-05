# ML-Sentinel: Production MLOps and Drift Detection Pipeline

## System Overview
ML-Sentinel is a fully automated MLOps infrastructure designed to manage the lifecycle of machine learning models in production. It centralizes feature engineering, monitors production data for statistical drift, and orchestrates automated retraining pipelines with zero-downtime deployments.

## Architecture Flow
1. **Feature Store (Feast):** Centralizes feature definitions. Ensures that offline training data and online serving data are strictly identical, preventing training-serving skew. Point-in-time correctness is enforced to avoid data leakage during temporal joins.
2. **Continuous Monitoring (Evidently AI):** Production inference payloads are logged and asynchronously analyzed. The system computes the Population Stability Index (PSI) for categorical features and runs Kolmogorov-Smirnov (KS) tests for numerical features to detect data drift.
3. **Automated Retraining (Kubeflow):** Upon detecting a drift threshold violation, a webhook triggers a Kubeflow pipeline. The DAG automatically extracts the latest data from the feature store, trains a new model, and logs artifacts to MLflow.
4. **Model Registry & Deployment:** If the newly trained model outperforms the production model on a holdout set, it is promoted in the MLflow registry and seamlessly rolled out via a shadow deployment strategy.

## Tech Stack
- **Feature Store:** Feast
- **Monitoring:** Evidently AI, Prometheus, Grafana
- **Orchestration:** Kubeflow Pipelines
- **Registry:** MLflow
