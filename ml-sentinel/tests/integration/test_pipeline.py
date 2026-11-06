import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mlflow.train import train


def test_pipeline_runs():
    run_id = train(
        data_path="feast_repo/data/driver_stats.parquet"
    )
    assert run_id is not None
