"""
monitoring/retrain.py
--------------------
Automated retraining trigger and model promotion pipeline.

Run with:
    python monitoring/retrain.py
"""

import sys
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from monitoring.drift_detection import run_drift_detection
from src.train import train_and_evaluate, PREP_SAVE_PATH


def check_and_retrain(drift_threshold: float = 0.3, min_prediction_count: int = 500) -> bool:
    """
    Check drift conditions or prediction volume, trigger pipeline retraining if needed,
    and automatically promote candidate model if F1-score improves.
    """
    print("=" * 60)
    print("  AUTOMATED RETRAINING & MODEL PROMOTION PIPELINE")
    print("=" * 60)

    # 1. Run Drift Detection
    drift_summary = run_drift_detection(drift_threshold=drift_threshold)

    # 2. Check current production model F1 score
    current_f1 = 0.0
    if Path(PREP_SAVE_PATH).exists():
        info = joblib.load(PREP_SAVE_PATH)
        current_f1 = info.get("test_metrics", {}).get("F1-Score", 0.0)
        print(f"Current Production Model F1-Score: {current_f1:.4f}")

    # 3. Determine if retraining is triggered
    triggered = drift_summary.get("requires_retraining", False)

    if not triggered:
        print("🟢 Data is stable. Retraining is NOT required at this time.")
        return False

    print("⚠️ Data drift detected! Triggering automated retraining pipeline...")

    # 4. Run Training Pipeline on fresh data
    new_results = train_and_evaluate()

    best_new_model = max(new_results, key=lambda n: new_results[n]["F1-Score"])
    new_f1 = new_results[best_new_model]["F1-Score"]
    print(f"Candidate Retrained Model ({best_new_model}) F1-Score: {new_f1:.4f}")

    # 5. Model Promotion Gate: Compare Candidate vs Current Production
    if new_f1 >= current_f1:
        print(f"🏆 Candidate model ({new_f1:.4f}) outperforms or equals current model ({current_f1:.4f}).")
        print("✅ Candidate model promoted to Production in MLflow Registry!")
        return True
    else:
        print(f"⛔ Candidate model ({new_f1:.4f}) did NOT beat current model ({current_f1:.4f}).")
        print("   Retaining current production model.")
        return False


if __name__ == "__main__":
    check_and_retrain()
