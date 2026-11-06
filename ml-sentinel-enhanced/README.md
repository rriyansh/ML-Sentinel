# ml-sentinel-enhanced

Enhanced MLOps pipeline with Feast feature store, MLflow model training with lineage & tuning, Evidently drift monitoring (JSON + HTML), Airflow orchestration with branching & quality gate, and FastAPI serving with health checks & API key.

## Structure

```
ml-sentinel-enhanced/
├── airflow/dags/ml_sentinel_dag.py       # Enhanced DAG with branching & quality gate
├── config/
│   ├── drift_config.yaml                 # Drift thresholds & feature lists
│   └── promotion_policy.json             # Model promotion rules (Champion/Challenger)
├── feast_repo/                           # Feast feature store repository
│   ├── feature_store.yaml                # Feast configuration with MLflow integration
│   ├── features.py                       # Feature definitions (Entities, Views)
│   └── data/driver_stats.parquet         # Toy data
├── mlflow/train.py                       # Enhanced training with Feast lineage & tuning
├── monitoring/drift_check.py             # Enhanced drift report (JSON + HTML)
├── serving/
│   ├── app.py                            # FastAPI with health checks & API key
│   └── schemas.py                        # Pydantic request/response models
├── requirements.txt
└── README.md
```

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Feature store (Feast):
   ```
   cd feast_repo
   feast apply
   ```

3. Train model with MLflow:
   ```
   python mlflow/train.py
   ```

4. Run drift detection:
   ```
   python monitoring/drift_check.py
   ```

5. Start serving API:
   ```
   uvicorn serving.app:app --reload
   ```

6. Run the Airflow DAG:
   ```
   export AIRFLOW_HOME=~/airflow
   airflow dags trigger ml_sentinel_pipeline
   ```
