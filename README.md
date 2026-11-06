# ML-Sentinel

Production MLOps pipeline with Feast feature store, MLflow model training, Evidently drift monitoring, Airflow orchestration, and FastAPI serving.

## Structure

```
ml-sentinel/
├── .github/workflows/ci-cd.yaml        # CI/CD pipeline (test, build, push)
├── airflow/dags/ml_sentinel_dag.py     # Production Airflow DAG
├── config/
│   ├── drift_config.yaml               # Drift thresholds & feature lists
│   ├── model_promotion_policy.json     # Champion/Challenger rules
│   └── logging.yaml                    # Centralized logging configuration
├── data/
│   ├── reference/driver_stats.parquet  # Reference dataset for drift comparison
│   └── current/                        # Directory for incoming live data
├── docker/
│   ├── Dockerfile.airflow              # Custom Airflow image with dependencies
│   ├── Dockerfile.serving              # FastAPI serving container
│   └── Dockerfile.train                # Training job container
├── feast_repo/
│   ├── feature_store.yaml              # Feast configuration
│   ├── features.py                     # Feature definitions (Entities, Views)
│   └── data/driver_stats.parquet       # Feature store data
├── mlflow/train.py                     # Training script with MLflow & Feast
├── monitoring/drift_check.py           # Evidently drift detection logic
├── serving/
│   ├── app.py                          # FastAPI application
│   └── schemas.py                      # Pydantic request/response models
├── src/ml_sentinel/
│   ├── __init__.py                     # Package init
│   ├── config.py                       # Configuration loader
│   ├── logger.py                       # Logger setup
│   └── utils.py                        # Shared utilities
├── tests/
│   ├── unit/
│   │   ├── test_drift_check.py
│   │   └── test_serving.py
│   └── integration/
│       └── test_pipeline.py
├── .gitignore
├── Makefile                            # Common commands (setup, test, run)
├── pyproject.toml                      # Project metadata & dependencies
├── requirements.txt                    # Pinned dependencies for Docker
└── README.md
```

## Usage

1. Install dependencies:
   ```
   cd ml-sentinel
   make setup
   ```

2. Train model:
   ```
   make train
   ```

3. Run drift detection:
   ```
   make drift
   ```

4. Start serving API:
   ```
   make serve
   ```

5. Run tests:
   ```
   make test
   ```
