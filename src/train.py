"""
src/train.py
-------------
Trains three classification models on the student performance dataset,
compares their performance, and saves the best model to disk.

Models trained:
    1. Logistic Regression  — a simple linear classifier (good baseline)
    2. Random Forest        — an ensemble of decision trees (handles non-linearity)
    3. XGBoost              — gradient boosting (often best in practice)

Run from the project root (with .venv activated):
    python src/train.py

Expected output:
    - A comparison table of accuracy, precision, recall, F1 for all 3 models
    - A confusion matrix for the best model
    - Saves  models/best_model.pkl  and  models/preprocessor_info.pkl
"""

import sys
import os
import io
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn/xgboost version warnings

# Force UTF-8 output on Windows to prevent encoding errors in terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Make sure Python can find the src/ module when run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

from src.data_preprocessing import (
    run_full_pipeline,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
)

# ------------------------------------------------------------------
# PATHS — all relative to the project root, never hardcoded deeply
# ------------------------------------------------------------------
DATA_PATH       = "data/raw/student-mat.csv"
MODEL_SAVE_PATH = "models/best_model.pkl"
PREP_SAVE_PATH  = "models/preprocessor_info.pkl"


# ------------------------------------------------------------------
# HELPER: Print a formatted comparison table
# ------------------------------------------------------------------
def print_comparison_table(results: dict) -> None:
    """
    Print a nicely formatted table comparing model metrics.

    Args:
        results: dict of {model_name: {metric_name: value}}
    """
    # Column headers
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    col_width = 12

    header = f"{'Model':<22}" + "".join(f"{m:>{col_width}}" for m in metrics)
    divider = "-" * len(header)

    print()
    print("=" * len(header))
    print("  MODEL COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print(divider)

    for model_name, scores in results.items():
        row = f"{model_name:<22}"
        for m in metrics:
            row += f"{scores[m]:>{col_width}.4f}"
        print(row)

    print(divider)
    print()


# ------------------------------------------------------------------
# HELPER: Print a confusion matrix with labels
# ------------------------------------------------------------------
def print_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """
    Print a labelled confusion matrix.

    A confusion matrix shows:
        - True Positives  (TPs): predicted Pass, actually Pass
        - True Negatives  (TNs): predicted Fail, actually Fail
        - False Positives (FPs): predicted Pass, actually Fail  ← "false alarm"
        - False Negatives (FNs): predicted Fail, actually Pass  ← "missed student"

    For this project, False Negatives (FNs) are more costly:
        Missing a student who will fail means we can't intervene early.

    Args:
        y_true:     True labels (from test set)
        y_pred:     Predicted labels
        model_name: Name to display in the header
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"{'='*50}")
    print(f"  Confusion Matrix -- {model_name}")
    print(f"{'='*50}")
    print(f"                    Predicted")
    print(f"                  Fail    Pass")
    print(f"  Actual  Fail  [{tn:>5}]  [{fp:>5}]")
    print(f"          Pass  [{fn:>5}]  [{tp:>5}]")
    print(f"{'='*50}")
    print(f"  True Negatives  (Fail correctly identified): {tn}")
    print(f"  False Positives (predicted Pass, actually Fail): {fp}  <- false alarm")
    print(f"  False Negatives (predicted Fail, actually Pass): {fn}  <- missed student!")
    print(f"  True Positives  (Pass correctly identified): {tp}")
    print()

    print("  Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fail", "Pass"]))


# ------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------------------------------------------
def train_and_evaluate(data_path: str = DATA_PATH) -> dict:
    """
    Run the full training pipeline:
        1. Preprocess data
        2. Train 3 models
        3. Evaluate and compare
        4. Save the best model

    Args:
        data_path: Path to the raw CSV file.

    Returns:
        Dictionary of results for all models.
    """

    # ── Step 1: Preprocess ──────────────────────────────────────────
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = run_full_pipeline(data_path)

    # ── Step 2: Define models ────────────────────────────────────────
    #
    # WHY these three models?
    #   • Logistic Regression: simplest possible classifier — good baseline.
    #     If a complex model barely beats it, it's not worth the complexity.
    #   • Random Forest: builds many decision trees and votes. Robust,
    #     handles feature interactions well, rarely overfits badly.
    #   • XGBoost: gradient boosting — iteratively corrects mistakes from
    #     previous trees. Often the best performer on tabular data.
    #
    # random_state=42 on all models: ensures reproducibility.
    # max_iter=1000 on LogReg: gives it enough iterations to converge.

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000,       # default 100 can fail to converge
            C=1.0,               # regularization strength (default)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,    # 100 trees in the forest
            random_state=42,
            n_jobs=-1,           # use all CPU cores
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,         # suppress XGBoost training output
        ),
    }

    # ── Step 3: Train, predict, and score ───────────────────────────
    results = {}
    trained_models = {}

    print("\nTraining models...\n")

    for model_name, model in models.items():
        print(f"  Training {model_name}...")

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict on the held-out test set
        y_pred = model.predict(X_test)

        # Compute metrics
        # - accuracy:  overall % correct
        # - precision: of all "Pass" predictions, how many were actually Pass?
        # - recall:    of all actual Passes, how many did we catch?
        # - F1:        harmonic mean of precision and recall (best single metric
        #              when classes are imbalanced, as they are here ~67/33)
        results[model_name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
        }

        trained_models[model_name] = (model, y_pred)

    # ── Step 4: Print comparison table ──────────────────────────────
    print_comparison_table(results)

    # ── Step 5: Find the best model (by F1-Score) ───────────────────
    #
    # WHY F1 and not accuracy?
    #   Our dataset is imbalanced: 67% Pass, 33% Fail.
    #   A model that predicts "Pass" for EVERYONE would get 67% accuracy!
    #   F1-Score penalises a model that ignores the minority class (Fail).

    best_name = max(results, key=lambda name: results[name]["F1-Score"])
    best_model, best_preds = trained_models[best_name]

    print(f"[BEST] Best model: {best_name}")
    print(f"   F1-Score: {results[best_name]['F1-Score']:.4f} | "
          f"Accuracy: {results[best_name]['Accuracy']:.4f}")

    # ── Step 6: Confusion matrix for the best model ─────────────────
    print_confusion_matrix(y_test, best_preds, best_name)

    # ── Step 7: Save the best model ─────────────────────────────────
    os.makedirs("models", exist_ok=True)  # create folder if it doesn't exist

    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"[OK] Best model saved to: {MODEL_SAVE_PATH}")

    # Save metadata about the preprocessor so the API knows what
    # columns to expect and what order they should be in.
    preprocessor_info = {
        "feature_columns": list(X_train.columns),   # exact column order after encoding
        "numeric_cols":    FEATURE_COLS,             # original feature list
        "categorical_cols": CATEGORICAL_COLS,        # which cols were one-hot encoded
        "target_col":       TARGET_COL,
        "best_model_name":  best_name,
        "test_metrics":     results[best_name],
    }
    joblib.dump(preprocessor_info, PREP_SAVE_PATH)
    print(f"[OK] Preprocessor info saved to: {PREP_SAVE_PATH}")

    return results


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Student Performance Prediction -- Model Training")
    print("=" * 55)

    results = train_and_evaluate(DATA_PATH)

    print("\nTraining complete! Check the models/ folder for saved artifacts.")
    print("   Next step: run  mlflow ui  to explore experiment tracking.\n")
