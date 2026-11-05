# ml-sentinel-basic

Basic MLOps pipeline with Feast feature store, MLflow model training, Evidently drift monitoring, Airflow orchestration, and FastAPI serving.

## Structure

```
ml-sentinel-basic/
├── airflow/dags/ml_sentinel_dag.py       # Airflow orchestration DAG
├── feast_repo/                           # Feast feature store repository
│   ├── feature_store.yaml                # Feast configuration
│   ├── features.py                       # Feature definitions (Entities, Views)
│   └── data/driver_stats.parquet         # Toy data
├── mlflow/train.py                       # Training script + MLflow logging
├── monitoring/drift_check.py             # Evidently drift detection logic
├── serving/app.py                        # FastAPI model serving
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
