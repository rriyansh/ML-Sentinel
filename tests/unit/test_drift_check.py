import pandas as pd
from monitoring.drift_check import check_drift

def test_no_drift():
    df = pd.DataFrame({"conv_rate": [0.1, 0.2], "acc_rate": [0.8, 0.9], "avg_daily_trips": [10, 20]})
    assert check_drift(df, df) is False

def test_drift():
    ref = pd.DataFrame({"conv_rate": [0.1, 0.2], "acc_rate": [0.8, 0.9], "avg_daily_trips": [10, 20]})
    curr = pd.DataFrame({"conv_rate": [0.9, 0.95], "acc_rate": [0.1, 0.2], "avg_daily_trips": [100, 200]})
    assert check_drift(ref, curr) is True
