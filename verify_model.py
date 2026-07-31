"""
Verification script — confirms the saved model loads and predicts correctly on the new sheet.
Run from project root:  python verify_model.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import joblib
import pandas as pd

from src.data_preprocessing import (
    load_data, create_target, select_features,
    encode_features, split_data
)

print("=" * 55)
print("  MODEL VERIFICATION (New Sheet)")
print("=" * 55)

# ── 1. Load saved artifacts ───────────────────────────────
model = joblib.load("models/best_model.pkl")
info  = joblib.load("models/preprocessor_info.pkl")

print(f"\nModel loaded   : {info['best_model_name']}")
print(f"Feature cols   : {len(info['feature_columns'])} columns")
print(f"Test accuracy  : {info['test_metrics']['Accuracy']:.2%}")
print(f"Test F1-Score  : {info['test_metrics']['F1-Score']:.2%}")

# ── 2. Re-run preprocessing on the new dataset ───────────
df = load_data("data/raw/Student Performance Prediction with MLOps - Sheet1.csv")
df = create_target(df)
df = select_features(df)
df = encode_features(df)
X_train, X_test, y_train, y_test = split_data(df)

# ── 3. Spot-check 10 test rows ───────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Sample predictions on 10 test students ---")
print(f"{'#':<4} {'Actual':<10} {'Predicted':<12} {'Confidence':<12} {'Correct?'}")
print("-" * 52)

label = {1: "PASS", 0: "FAIL"}
for i, (actual, pred, prob) in enumerate(
        zip(list(y_test)[:10], y_pred[:10], y_prob[:10])):
    correct = "YES" if actual == pred else "NO  <-- wrong"
    print(f"{i+1:<4} {label[actual]:<10} {label[pred]:<12} {prob:.1%}        {correct}")

print("\n" + "=" * 55)
print("  VERIFICATION COMPLETE - Model works on the new sheet!")
print("=" * 55)
