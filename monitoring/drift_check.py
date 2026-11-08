import pandas as pd
import json
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns, TestNumberOfDriftedColumns
from src.ml_sentinel.config import config
from src.ml_sentinel.logger import setup_logger

logger = setup_logger("drift_check")

def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    features = config.drift["features_to_monitor"]
    threshold = config.drift["threshold"]

    suite = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=threshold),
        TestNumberOfDriftedColumns(lt=len(features) // 2),
    ])

    suite.run(reference_data=reference_df[features], current_data=current_df[features])

    # Save reports
    suite.save_html("monitoring/drift_report.html")
    with open("monitoring/drift_summary.json", "w") as f:
        json.dump(suite.json(), f)

    # The suite fails if the share of drifted columns >= threshold
    drift_detected = not suite.as_dict()["summary"]["all_passed"]
    logger.info(f"Drift check completed. Drift detected: {drift_detected}")
    return drift_detected
