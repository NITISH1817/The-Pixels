"""
tests/test_preprocessing.py
----------------------------
Unit tests for data preprocessing pipeline functions in src/data_preprocessing.py.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_data,
    create_target,
    select_features,
    encode_features,
    split_data,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)


# ------------------------------------------------------------------
# FIXTURE: Synthetic raw dataframe matching Sheet1 structure
# ------------------------------------------------------------------
@pytest.fixture
def raw_sample_df():
    data = {
        "Gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        "Age": [20, 22, 19, 21, 23, 20, 21, 19],
        "Study_Hours_per_Week": [15, 5, 25, 12, 8, 18, 30, 6],
        "Attendance_Rate": ["95.0%", "60.0%", "99.0%", "85.0%", "70.0%", "92.0%", "100.0%", "65.0%"],
        "Past_Exam_Scores": [85, 55, 92, 75, 65, 82, 96, 58],
        "Assignment_Submission_Rate": ["98.0%", "50.0%", "100.0%", "80.0%", "65.0%", "90.0%", "100.0%", "60.0%"],
        "Quiz_Average": [88, 60, 95, 78, 68, 85, 98, 62],
        "Previous_Failures": [0, 2, 0, 0, 1, 0, 0, 1],
        "Internet_Access": ["Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
        "Parents_Support": ["High", "Low", "High", "Medium", "Low", "High", "High", "Medium"],
        "Parental_Education": ["Bachelor's", "High School", "Master's", "Bachelor's", "High School", "Bachelor's", "PhD", "High School"],
        "Family_Income": ["Medium", "Low", "High", "Medium", "Low", "High", "High", "Medium"],
        "Sleep_Hours": [7, 5, 8, 6, 6, 7, 8, 5],
        "Screen_Time": [3, 6, 2, 4, 5, 3, 1, 7],
        "Stress_Level": ["Medium", "High", "Low", "Medium", "High", "Medium", "Low", "High"],
        "Motivation_Level": ["High", "Low", "High", "Medium", "Low", "High", "High", "Low"],
        "Class_Participation": ["High", "Low", "High", "Medium", "Low", "Medium", "High", "Low"],
        "Extracurricular_Activities": [2, 0, 3, 1, 0, 2, 4, 1],
        "School_Support": ["High", "Medium", "High", "Medium", "Low", "High", "High", "Medium"],
        "Travel_Time": [30, 60, 15, 45, 90, 20, 10, 50],
        "Pass_Fail": ["Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail"],
    }
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# TESTS: 1. create_target
# ------------------------------------------------------------------
class TestCreateTarget:

    def test_pass_fail_rule_correct(self, raw_sample_df):
        df = create_target(raw_sample_df)
        assert df.loc[0, "Pass_Fail"] == 1  # Pass -> 1
        assert df.loc[1, "Pass_Fail"] == 0  # Fail -> 0

    def test_target_column_is_binary(self, raw_sample_df):
        df = create_target(raw_sample_df)
        unique_vals = set(df["Pass_Fail"].unique())
        assert unique_vals.issubset({0, 1})

    def test_row_count_unchanged(self, raw_sample_df):
        df = create_target(raw_sample_df)
        assert len(df) == len(raw_sample_df)


# ------------------------------------------------------------------
# TESTS: 2. select_features
# ------------------------------------------------------------------
class TestSelectFeatures:

    def test_returns_only_expected_columns(self, raw_sample_df):
        df = create_target(raw_sample_df)
        df_selected = select_features(df)
        expected = set(FEATURE_COLS + [TARGET_COL])
        assert set(df_selected.columns) == expected

    def test_column_count_is_correct(self, raw_sample_df):
        df = create_target(raw_sample_df)
        df_selected = select_features(df)
        assert len(df_selected.columns) == len(FEATURE_COLS) + 1

    def test_no_rows_dropped(self, raw_sample_df):
        df = create_target(raw_sample_df)
        df_selected = select_features(df)
        assert len(df_selected) == len(df)


# ------------------------------------------------------------------
# TESTS: 3. split_data
# ------------------------------------------------------------------
class TestSplitData:

    def test_train_test_proportion_approximate(self, raw_sample_df):
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)
        assert len(X_train) == 6
        assert len(X_test) == 2

    def test_no_overlap_between_train_and_test(self, raw_sample_df):
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert train_indices.isdisjoint(test_indices)

    def test_reproducibility_same_split_every_time(self, raw_sample_df):
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)
        X_tr1, X_te1, _, _ = split_data(df, random_state=42)
        X_tr2, X_te2, _, _ = split_data(df, random_state=42)
        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_frame_equal(X_te1, X_te2)
