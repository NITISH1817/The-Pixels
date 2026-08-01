"""
src/train.py
-------------
Trains three classification models on the student performance dataset,
compares their performance, logs everything to MLflow, and registers the best model.

Features:
    - MLflow Experiment Tracking (params, metrics, artifacts)
    - MLflow Model Registry integration
    - Expanded 20-feature dataset (Sheet1)
    - StandardScaler + Pipeline
    - GridSearchCV hyperparameter tuning

Run from the project root (with .venv activated):
    python src/train.py
"""

import sys
import os
import warnings
import io
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

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
EXPERIMENT_NAME = "Student_Performance_Prediction"
REGISTERED_MODEL_NAME = "student_performance_model"


# ------------------------------------------------------------------
# HELPER: Save Confusion Matrix Image for MLflow Artifacts
# ------------------------------------------------------------------
def save_confusion_matrix_plot(y_true, y_pred, model_name: str, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"],
                yticklabels=["Fail", "Pass"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ------------------------------------------------------------------
# HELPER: Save Feature Importance Plot for MLflow Artifacts
# ------------------------------------------------------------------
def save_feature_importance_plot(model, feature_names, output_path: str):
    try:
        inner_model = model.named_steps["model"]
        if hasattr(inner_model, "feature_importances_"):
            importances = inner_model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
            plt.figure(figsize=(8, 5))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
            plt.title("Top 10 Feature Importances")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
    except Exception:
        pass


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
# MAIN TRAINING FUNCTION WITH MLFLOW LOGGING
# ------------------------------------------------------------------
def train_and_evaluate(data_path: str = DATA_PATH) -> dict:

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = run_full_pipeline(data_path)
    print(f"\nFeature matrix shape: {X_train.shape} train / {X_test.shape} test")

    # Estimate class weight ratio for XGBoost
    n_pass = int(y_train.sum())
    n_fail = int(len(y_train) - n_pass)
    xgb_scale = round(n_pass / n_fail, 2)

    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(random_state=42, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)),
        ]),
    }

    param_grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth":    [10, 15, None],
        },
        "XGBoost": {
            "model__n_estimators":  [150, 250],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":     [4, 6],
        },
    }

    results        = {}
    trained_models = {}

    print("\nTraining + tuning models with MLflow tracking...\n")

    os.makedirs("models", exist_ok=True)

    for model_name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=model_name):
            print(f"  Tuning & Logging {model_name} to MLflow...")

            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=5,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_
            cv_f1 = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring="f1_macro").mean()

            y_pred = best_pipeline.predict(X_test)

            acc   = accuracy_score(y_test, y_pred)
            prec  = precision_score(y_test, y_pred, zero_division=0)
            rec   = recall_score(y_test, y_pred, zero_division=0)
            f1    = f1_score(y_test, y_pred, zero_division=0)

            # Log Hyperparameters to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("dataset_size", len(X_train) + len(X_test))
            mlflow.log_param("features_count", X_train.shape[1])

            # Log Metrics to MLflow
            mlflow.log_metric("cv_f1_macro", cv_f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log Artifacts (Plots)
            cm_plot_path = f"models/cm_{model_name.lower().replace(' ', '_')}.png"
            save_confusion_matrix_plot(y_test, y_pred, model_name, cm_plot_path)
            mlflow.log_artifact(cm_plot_path)

            fi_plot_path = f"models/fi_{model_name.lower().replace(' ', '_')}.png"
            save_feature_importance_plot(best_pipeline, X_train.columns, fi_plot_path)
            if os.path.exists(fi_plot_path):
                mlflow.log_artifact(fi_plot_path)

            # Log Model to MLflow using cloudpickle serialization
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                best_pipeline,
                name="model",
                signature=signature,
                serialization_format="cloudpickle"
            )

            results[model_name] = {
                "CV-F1 (mean)": cv_f1,
                "Accuracy":     acc,
                "Precision":    prec,
                "Recall":       rec,
                "F1-Score":     f1,
            }
            trained_models[model_name] = (best_pipeline, y_pred)

    # Print comparison
    print_comparison_table(results)

    # Selected model (dynamically pick highest accuracy & F1 score)
    best_name = max(results.keys(), key=lambda k: (results[k]["Accuracy"], results[k]["F1-Score"]))
    best_model, best_preds = trained_models[best_name]

    print(f"[BEST] Winning Model: {best_name}")
    print(f"   Test F1-Score : {results[best_name]['F1-Score']:.4f}")
    print(f"   Test Accuracy : {results[best_name]['Accuracy']:.4f}")

    # Register Winning Model in MLflow Model Registry
    with mlflow.start_run(run_name=f"REGISTERED_{best_name}") as run:
        signature = infer_signature(X_train, best_preds)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle"
        )
        print(f"[MLflow] Registered model '{REGISTERED_MODEL_NAME}' in MLflow Model Registry!")

    # Save local pkl artifacts as backup
    joblib.dump(best_model, MODEL_SAVE_PATH)
    preprocessor_info = {
        "feature_columns":  list(X_train.columns),
        "numeric_cols":     FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_col":       TARGET_COL,
        "best_model_name":  best_name,
        "test_metrics":     results[best_name],
    }
    joblib.dump(preprocessor_info, PREP_SAVE_PATH)
    print(f"[OK] Local model saved to: {MODEL_SAVE_PATH}")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Student Performance Prediction -- MLflow Training")
    print("=" * 55)

    results = train_and_evaluate(DATA_PATH)

    print("\nTraining complete! All runs logged to MLflow.")
    print("Run  mlflow ui  to view experiment dashboard!\n")
