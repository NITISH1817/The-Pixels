"""
src/train.py
-------------
Trains three classification models on the student performance dataset,
compares their performance, and saves the best model to disk.

Improvements over v1:
    - Expanded from 12 to 24 features (adds higher, goout, Walc, famrel, etc.)
    - Uses sklearn Pipeline with StandardScaler for proper feature scaling
    - Hyperparameter tuning via GridSearchCV (cross-validation)
    - Cross-validated F1 scores for more reliable model comparison

Run from the project root (with .venv activated):
    python src/train.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from src.data_preprocessing import (
    run_full_pipeline, FEATURE_COLS, CATEGORICAL_COLS, TARGET_COL,
)

DATA_PATH       = "data/raw/Student Performance Prediction with MLOps - Sheet1.csv"
MODEL_SAVE_PATH = "models/best_model.pkl"
PREP_SAVE_PATH  = "models/preprocessor_info.pkl"


# ------------------------------------------------------------------
# HELPER: Print comparison table
# ------------------------------------------------------------------
def print_comparison_table(results: dict) -> None:
    metrics   = ["CV-F1 (mean)", "Accuracy", "Precision", "Recall", "F1-Score"]
    col_width = 14
    header    = f"{'Model':<22}" + "".join(f"{m:>{col_width}}" for m in metrics)
    divider   = "-" * len(header)

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
# HELPER: Print confusion matrix
# ------------------------------------------------------------------
def print_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"{'='*55}")
    print(f"  Confusion Matrix -- {model_name}")
    print(f"{'='*55}")
    print(f"                    Predicted")
    print(f"                  Fail    Pass")
    print(f"  Actual  Fail  [{tn:>5}]  [{fp:>5}]")
    print(f"          Pass  [{fn:>5}]  [{tp:>5}]")
    print(f"{'='*55}")
    print(f"  True Negatives  (correctly identified Fail): {tn}")
    print(f"  False Positives (said Pass, actually Fail) : {fp}  <- false alarm")
    print(f"  False Negatives (said Fail, actually Pass) : {fn}  <- missed student!")
    print(f"  True Positives  (correctly identified Pass): {tp}")
    print()
    print("  Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fail", "Pass"]))


# ------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------------------------------------------
def train_and_evaluate(data_path: str = DATA_PATH) -> dict:
    """
    Full training pipeline with scaling, cross-validation, and tuning.
    """

    # ── Step 1: Preprocess ──────────────────────────────────────────
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = run_full_pipeline(data_path)
    print(f"\nFeature matrix shape: {X_train.shape} train / {X_test.shape} test")

    # ── Step 2: Define Pipelines ─────────────────────────────────────
    #
    # WHY use a Pipeline?
    #   A Pipeline bundles preprocessing (scaling) + model into ONE object.
    #   This means the scaler is fitted ONLY on training data (no data leakage).
    #
    # WHY class_weight='balanced'?
    #   Our data has 67% Pass / 33% Fail. Without balancing, the model learns
    #   that predicting "Pass" for everyone is an easy win. Balanced weighting
    #   tells it: "Catching a Fail is twice as important as catching a Pass"
    #   because there are half as many Fail examples.

    # Ratio used for XGBoost: negative(Fail)/positive(Pass) = 130/265
    # XGBoost uses scale_pos_weight to up-weight the minority class
    n_pass = int(y_train.sum())
    n_fail = int(len(y_train) - n_pass)
    xgb_scale = round(n_pass / n_fail, 2)  # >1 means Pass is majority

    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight="balanced",  # penalise misclassifying Fail more
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",  # same balancing logic
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                scale_pos_weight=xgb_scale,  # balances Pass vs Fail weight
            )),
        ]),
    }

    # ── Step 3: Hyperparameter grids ────────────────────────────────
    #
    # WHY GridSearchCV?
    #   Instead of guessing the best hyperparameters, we try multiple
    #   combinations and use 5-fold cross-validation to pick the winner.
    #   "5-fold CV" = split training data into 5 parts, train on 4, test on 1,
    #   repeat 5 times and average — gives a much more reliable score.
    #
    # Note: Pipeline param names use format "stepname__paramname"

    # WHY f1_macro?
    #   `scoring="f1"` only measures F1 for the *Pass* class (label=1).
    #   A model that predicts "Pass" for everyone gets a perfect f1 score!
    #   `f1_macro` averages F1 across BOTH classes equally, so the model
    #   is forced to catch failing students too.

    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
        "XGBoost": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":     [3, 5],
        },
    }

    # ── Step 4: Train, tune, and score ───────────────────────────────
    results        = {}
    trained_models = {}

    print("\nTraining + tuning models (this takes ~30-60 seconds)...\n")

    for model_name, pipeline in pipelines.items():
        print(f"  Tuning {model_name}...")

        # GridSearchCV tries every combo in param_grid with 5-fold CV
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=5,
            scoring="f1_macro",   # optimise equally for BOTH Pass and Fail
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        print(f"    Best params: {grid_search.best_params_}")

        # Cross-validated macro-F1 on training data (5-fold)
        cv_f1 = cross_val_score(best_pipeline, X_train, y_train,
                                cv=5, scoring="f1_macro").mean()

        # Final evaluation on the held-out test set
        y_pred = best_pipeline.predict(X_test)

        results[model_name] = {
            "CV-F1 (mean)": cv_f1,
            "Accuracy":     accuracy_score(y_test, y_pred),
            "Precision":    precision_score(y_test, y_pred, zero_division=0),
            "Recall":       recall_score(y_test, y_pred, zero_division=0),
            "F1-Score":     f1_score(y_test, y_pred, zero_division=0),
        }
        trained_models[model_name] = (best_pipeline, y_pred)

    # ── Step 5: Print comparison table ───────────────────────────────
    print_comparison_table(results)

    # ── Step 6: Find best model by test F1 ───────────────────────────
    best_name = max(results, key=lambda n: results[n]["F1-Score"])
    best_model, best_preds = trained_models[best_name]

    print(f"[BEST] Best model: {best_name}")
    print(f"   Test F1-Score : {results[best_name]['F1-Score']:.4f}")
    print(f"   Test Accuracy : {results[best_name]['Accuracy']:.4f}")

    # ── Step 7: Confusion matrix ─────────────────────────────────────
    print()
    print_confusion_matrix(y_test, best_preds, best_name)

    # ── Step 8: Feature importances (if available) ───────────────────
    try:
        inner_model = best_model.named_steps["model"]
        if hasattr(inner_model, "feature_importances_"):
            importances = inner_model.feature_importances_
            feat_names  = X_train.columns
            top = sorted(zip(feat_names, importances),
                         key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 most important features:")
            for feat, imp in top:
                bar = "#" * int(imp * 200)
                print(f"  {feat:<25} {imp:.4f}  {bar}")
            print()
    except Exception:
        pass  # Logistic Regression uses coefficients, not importances — skip

    # ── Step 9: Save ─────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"[OK] Best model (pipeline) saved to: {MODEL_SAVE_PATH}")

    preprocessor_info = {
        "feature_columns":  list(X_train.columns),
        "numeric_cols":     FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
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

    print("\nTraining complete! Check models/ for saved artifacts.")
    print("Next step: run  python src/train.py  again to see if results change.\n")
