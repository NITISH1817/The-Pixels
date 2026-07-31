"""
monitoring/logger.py
--------------------
Utility module for automatic prediction logging in CSV and JSONL formats.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Configure directory paths using pathlib
MONITORING_DIR = Path(__file__).resolve().parent
LOGS_DIR = MONITORING_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSONL_LOG_PATH = LOGS_DIR / "predictions.jsonl"
CSV_LOG_PATH = LOGS_DIR / "predictions.csv"


def log_prediction(
    input_features: Dict[str, Any],
    prediction: str,
    success_probability: float,
    risk_score: float,
    risk_level: str,
    model_version: str = "v2.0.0"
) -> None:
    """
    Log single prediction event to both JSONL and CSV files.
    """
    timestamp = datetime.now().isoformat()

    log_record = {
        "timestamp": timestamp,
        "model_version": model_version,
        "prediction": prediction,
        "success_probability": success_probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "input_features": input_features
    }

    # 1. Append to JSONL
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    # 2. Append to CSV
    csv_exists = CSV_LOG_PATH.exists()
    
    # Flatten input features for CSV format
    flat_record = {
        "timestamp": timestamp,
        "model_version": model_version,
        "prediction": prediction,
        "success_probability": success_probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        **input_features
    }

    fieldnames = list(flat_record.keys())

    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(flat_record)
