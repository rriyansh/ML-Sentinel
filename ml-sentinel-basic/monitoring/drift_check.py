import os

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_PATH = os.getenv(
    "REFERENCE_PATH", "feast_repo/data/driver_stats.parquet"
)
CURRENT_PATH = os.getenv("CURRENT_PATH", "data/current.parquet")
REPORT_PATH = os.getenv("REPORT_PATH", "drift_report.html")

COLUMNS = ["conv_rate", "acc_rate", "avg_daily_trips"]


def check_drift(reference_path: str = REFERENCE_PATH, current_path: str = CURRENT_PATH) -> bool:
    reference = pd.read_parquet(reference_path)[COLUMNS]
    current = pd.read_parquet(current_path)[COLUMNS]

    column_mapping = ColumnMapping(numerical_features=COLUMNS)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    report.save_html(REPORT_PATH)

    result = report.as_dict()
    drift_detected = any(
        metric.get("value", {}).get("dataset_drift", False)
        for metric in result.get("metrics", [])
    )

    print(f"Drift report saved to {REPORT_PATH}")
    print(f"Dataset drift detected: {drift_detected}")
    return drift_detected


if __name__ == "__main__":
    check_drift()
