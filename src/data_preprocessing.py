"""
src/data_preprocessing.py
--------------------------
Reusable data preprocessing functions for the new Student Performance dataset
(Student Performance Prediction with MLOps - Sheet1.csv).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Categorical & Numeric Features for New Dataset
# ------------------------------------------------------------------
CATEGORICAL_COLS = [
    "Gender",
    "Internet_Access",
    "Parents_Support",
    "Parental_Education",
    "Family_Income",
    "Stress_Level",
    "Motivation_Level",
    "Class_Participation",
    "School_Support",
]

NUMERIC_COLS = [
    "Age",
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Assignment_Submission_Rate",
    "Quiz_Average",
    "Previous_Failures",
    "Sleep_Hours",
    "Screen_Time",
    "Extracurricular_Activities",
    "Travel_Time",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "Pass_Fail"


# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load the new CSV file.
    Cleans percentage strings (e.g. '95.0%' -> 95.0) in numeric columns.
    """
    df = pd.read_csv(path)

    # Clean percentage columns if they are strings with '%'
    for col in ["Attendance_Rate", "Assignment_Submission_Rate"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.rstrip("%").astype(float)

    print(f"[load_data] Loaded {len(df)} rows, {len(df.columns)} columns from '{path}'")
    return df


# ------------------------------------------------------------------
# 2. CREATE TARGET
# ------------------------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Pass_Fail string ('Pass'/'Fail') into binary integer (1/0).
    """
    df = df.copy()

    # Convert 'Pass' -> 1, 'Fail' -> 0
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().map({"Pass": 1, "Fail": 0})

    pass_count = df[TARGET_COL].sum()
    fail_count = len(df) - pass_count
    print(f"[create_target] Pass: {pass_count} ({pass_count/len(df)*100:.1f}%)  "
          f"Fail: {fail_count} ({fail_count/len(df)*100:.1f}%)")

    return df


# ------------------------------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------------------------------
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only expected feature columns + target column.
    """
    cols_to_keep = FEATURE_COLS + [TARGET_COL]

    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        raise ValueError(f"[select_features] Missing columns in DataFrame: {missing}")

    df = df[cols_to_keep].copy()
    print(f"[select_features] Kept {len(FEATURE_COLS)} features + target. Shape: {df.shape}")
    return df


# ------------------------------------------------------------------
# 4. ENCODE FEATURES
# ------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features.
    """
    df = df.copy()

    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True,
        dtype=int
    )

    print(f"[encode_features] After encoding: {df_encoded.shape[1]} columns (was {df.shape[1]})")
    return df_encoded


# ------------------------------------------------------------------
# 5. SPLIT DATA
# ------------------------------------------------------------------
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into 80% train and 20% test.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"[split_data] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# FULL PIPELINE WRAPPER
# ------------------------------------------------------------------
def run_full_pipeline(data_path: str):
    """
    Execute full pipeline from CSV path to train/test splits.
    """
    print("=" * 55)
    print("  Running full preprocessing pipeline")
    print("=" * 55)

    df = load_data(data_path)
    df = create_target(df)
    df = select_features(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("=" * 55)
    print("  Pipeline complete!")
    print("=" * 55)

    return X_train, X_test, y_train, y_test
