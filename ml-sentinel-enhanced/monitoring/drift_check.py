import json
import os

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

CONFIG_PATH = os.getenv("DRIFT_CONFIG_PATH", "config/drift_config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check_drift(config_path: str = CONFIG_PATH) -> dict:
    config = load_config(config_path)
    drift_cfg = config["drift"]

    reference = pd.read_parquet(drift_cfg["reference_dataset"])[drift_cfg["features"]]
    current = pd.read_parquet(drift_cfg["current_dataset"])[drift_cfg["features"]]

    column_mapping = ColumnMapping(numerical_features=drift_cfg["features"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    result = report.as_dict()

    dataset_drift = any(
        metric.get("value", {}).get("dataset_drift", False)
        for metric in result.get("metrics", [])
    )

    output = {
        "dataset_drift_detected": dataset_drift,
        "psi_threshold": drift_cfg["psi_threshold"],
        "ks_threshold": drift_cfg["ks_threshold"],
        "features": drift_cfg["features"],
    }

    os.makedirs("monitoring_reports", exist_ok=True)
    with open("monitoring_reports/drift_report.json", "w") as f:
        json.dump(output, f, indent=2)
    report.save_html("monitoring_reports/drift_report.html")

    print(json.dumps(output, indent=2))
    print("Drift report saved to monitoring_reports/drift_report.json and .html")
    return output


if __name__ == "__main__":
    check_drift()
