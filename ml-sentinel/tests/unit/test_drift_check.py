import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.ml_sentinel.config import load_config


def test_config_loading():
    config = load_config("drift_config.yaml")
    assert "drift" in config
    assert "features" in config["drift"]
    assert "psi_threshold" in config["drift"]


def test_drift_check_structure():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "drift_config.yaml")
    config = load_config("drift_config.yaml")
    features = config["drift"]["features"]
    assert "conv_rate" in features
    assert "acc_rate" in features
    assert "avg_daily_trips" in features


def test_reference_data_exists():
    ref_path = "data/reference/driver_stats.parquet"
    assert os.path.exists(ref_path), f"Reference data not found at {ref_path}"
    df = pd.read_parquet(ref_path)
    assert not df.empty


def test_serving_schema():
    from serving.schemas import DriverFeatures, PredictionResponse

    features = DriverFeatures(conv_rate=0.5, acc_rate=0.9, avg_daily_trips=20)
    assert features.conv_rate == 0.5
    assert features.acc_rate == 0.9
    assert features.avg_daily_trips == 20
