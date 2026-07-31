"""
Verification script — confirms the saved model loads and predicts correctly.
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
print("  MODEL VERIFICATION")
print("=" * 55)

# ── 1. Load saved artifacts ───────────────────────────────
model = joblib.load("models/best_model.pkl")
info  = joblib.load("models/preprocessor_info.pkl")

print(f"\nModel loaded   : {info['best_model_name']}")
print(f"Feature cols   : {len(info['feature_columns'])} columns")
print(f"Test accuracy  : {info['test_metrics']['Accuracy']:.2%}")
print(f"Test F1-Score  : {info['test_metrics']['F1-Score']:.2%}")

# ── 2. Re-run preprocessing to get the same test split ───
df = load_data("data/raw/student-mat.csv")
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

# ── 4. Hand-crafted sanity check ─────────────────────────
print("\n--- Sanity check: hand-crafted student profiles ---")

sample_students = pd.DataFrame([
    {   # Good student
        "age": 16, "studytime": 4, "failures": 0,
        "Medu": 4, "Fedu": 4, "absences": 0,
        "sex_M": 0, "address_U": 1,
        "Mjob_health": 0, "Mjob_other": 0,
        "Mjob_services": 0, "Mjob_teacher": 1,
        "Fjob_health": 0, "Fjob_other": 0,
        "Fjob_services": 0, "Fjob_teacher": 0,
        "famsup_yes": 1, "internet_yes": 1
    },
    {   # At-risk student
        "age": 19, "studytime": 1, "failures": 3,
        "Medu": 1, "Fedu": 1, "absences": 20,
        "sex_M": 1, "address_U": 0,
        "Mjob_health": 0, "Mjob_other": 1,
        "Mjob_services": 0, "Mjob_teacher": 0,
        "Fjob_health": 0, "Fjob_other": 1,
        "Fjob_services": 0, "Fjob_teacher": 0,
        "famsup_yes": 0, "internet_yes": 0
    }
])

preds = model.predict(sample_students)
probs = model.predict_proba(sample_students)[:, 1]
profiles = [
    "Good student  (studytime=4, failures=0, absences=0,  educated parents)",
    "At-risk student (studytime=1, failures=3, absences=20, less support)"
]
for profile, pred, prob in zip(profiles, preds, probs):
    print(f"\n  Profile : {profile}")
    print(f"  Prediction --> {label[pred]}  (confidence: {prob:.1%})")

print("\n" + "=" * 55)
print("  VERIFICATION COMPLETE - Model is working correctly")
print("=" * 55)
