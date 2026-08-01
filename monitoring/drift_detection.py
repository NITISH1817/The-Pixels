"""
monitoring/drift_detection.py
------------------------------
Data & Feature Drift Detection script using Evidently AI.

Run with:
    python monitoring/drift_detection.py
"""

import json
from pathlib import Path
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
    except ImportError:
        try:
            from evidently import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            Report = None
            DataDriftPreset = None

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Student Performance Prediction with MLOps - Sheet1.csv"
PREDICTION_LOGS_PATH = ROOT_DIR / "monitoring" / "logs" / "predictions.csv"
REPORTS_DIR = ROOT_DIR / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HTML_REPORT_PATH = REPORTS_DIR / "drift_report.html"
JSON_REPORT_PATH = REPORTS_DIR / "drift_summary.json"


def run_drift_detection(drift_threshold: float = 0.3) -> dict:
    """
    Compare reference training dataset vs current prediction logs and generate drift report.
    """
    print("=" * 55)
    print("  Evidently AI — Data Drift Detection")
    print("=" * 55)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found at {RAW_DATA_PATH}")

    # 1. Load Reference Data
    ref_df = pd.read_csv(RAW_DATA_PATH)
    # Strip % signs from reference data if present
    for col in ["Attendance_Rate", "Assignment_Submission_Rate"]:
        if col in ref_df.columns and ref_df[col].dtype == object:
            ref_df[col] = ref_df[col].astype(str).str.rstrip("%").astype(float)

    # Drop target column from reference feature set
    if "Pass_Fail" in ref_df.columns:
        ref_df = ref_df.drop(columns=["Pass_Fail"])

    # 2. Load Current Production Prediction Data
    if not PREDICTION_LOGS_PATH.exists():
        print(f"⚠️  No prediction logs found at {PREDICTION_LOGS_PATH}.")
        print("   Generating synthetic current dataset for drift check demonstration...")
        curr_df = ref_df.sample(frac=0.5, random_state=42).copy()
        # Simulate slight drift in attendance and study hours
        curr_df["Attendance_Rate"] = curr_df["Attendance_Rate"] * 0.85
        curr_df["Study_Hours_per_Week"] = curr_df["Study_Hours_per_Week"] * 0.70
    else:
        curr_df = pd.read_csv(PREDICTION_LOGS_PATH)
        # Select common feature columns between reference and prediction logs
        common_cols = [c for c in ref_df.columns if c in curr_df.columns]
        curr_df = curr_df[common_cols]

    # Align columns
    common_cols = [c for c in ref_df.columns if c in curr_df.columns]
    ref_df = ref_df[common_cols]
    curr_df = curr_df[common_cols]

    print(f"Reference rows: {len(ref_df)} | Current rows: {len(curr_df)}")

    # 3. Generate Evidently Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    # 4. Save HTML Report
    report.save_html(str(HTML_REPORT_PATH))
    print(f"✅ HTML Drift Report saved to: {HTML_REPORT_PATH}")

    # 5. Extract JSON Summary
    json_result = report.as_dict()
    metrics_summary = json_result["metrics"][0]["result"]

    dataset_drift = metrics_summary["dataset_drift"]
    drift_share = metrics_summary["drift_share"]
    number_of_drifted_columns = metrics_summary["number_of_drifted_columns"]

    summary = {
        "dataset_drift_detected": dataset_drift,
        "drift_share": drift_share,
        "number_of_drifted_columns": number_of_drifted_columns,
        "threshold": drift_threshold,
        "requires_retraining": drift_share >= drift_threshold or dataset_drift
    }

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ JSON Summary saved to: {JSON_REPORT_PATH}")
    print(f"   Dataset Drift Detected : {dataset_drift}")
    print(f"   Drifted Share          : {drift_share:.2%}")
    print(f"   Requires Retraining    : {summary['requires_retraining']}")

    return summary


if __name__ == "__main__":
    run_drift_detection()
