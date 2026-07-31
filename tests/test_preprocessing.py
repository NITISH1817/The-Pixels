"""
tests/test_preprocessing.py
----------------------------
pytest tests for src/data_preprocessing.py

Run from project root (with .venv activated):
    pytest tests/test_preprocessing.py -v

Why test preprocessing?
    Data bugs are silent — wrong label encoding, accidental column drops, or
    a bad train/test ratio won't crash your code but WILL ruin your model.
    These tests catch those bugs automatically.
"""

import pytest
import pandas as pd
import numpy as np

# Import the functions we want to test
from src.data_preprocessing import (
    create_target,
    select_features,
    encode_features,
    split_data,
    FEATURE_COLS,
    TARGET_COL,
)


# ------------------------------------------------------------------
# FIXTURES — shared test data that multiple tests can reuse
# ------------------------------------------------------------------

@pytest.fixture
def raw_sample_df():
    """
    A small hand-crafted DataFrame that mimics the raw student-mat.csv format.
    Using synthetic data means tests run fast and don't depend on the real file.
    """
    return pd.DataFrame({
        # Columns we will USE as features
        "age":       [15, 16, 17, 15, 18, 16, 15, 17],
        "sex":       ["F", "M", "F", "M", "F", "M", "F", "M"],
        "address":   ["U", "R", "U", "U", "R", "U", "R", "U"],
        "studytime": [2, 1, 3, 2, 1, 4, 2, 3],
        "failures":  [0, 1, 0, 2, 0, 0, 1, 0],
        "Medu":      [4, 2, 3, 1, 4, 3, 2, 4],
        "Fedu":      [4, 1, 3, 2, 3, 4, 2, 3],
        "Mjob":      ["teacher", "at_home", "other", "services", "health",
                      "teacher", "at_home", "other"],
        "Fjob":      ["services", "other", "teacher", "other", "other",
                      "services", "teacher", "other"],
        "famsup":    ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "internet":  ["yes", "yes", "no", "yes", "no", "yes", "yes", "no"],
        "absences":  [0, 4, 2, 6, 0, 2, 8, 0],
        # Grade columns that should be DROPPED after target creation
        "G1":        [10, 7, 14, 5, 15, 12, 8, 11],
        "G2":        [11, 8, 13, 6, 14, 13, 9, 12],
        # G3 is used ONLY to create the target label, then dropped
        "G3":        [12, 8, 15, 5, 16, 11, 9, 10],
        # Extra columns we DON'T use — should be dropped by select_features()
        "school":    ["GP"] * 8,
        "famsize":   ["GT3"] * 8,
        "Pstatus":   ["T"] * 8,
    })


# ------------------------------------------------------------------
# TEST 1: create_target()
# ------------------------------------------------------------------

class TestCreateTarget:

    def test_pass_fail_rule_correct(self, raw_sample_df):
        """
        G3 >= 10 should produce pass_fail = 1.
        G3 <  10 should produce pass_fail = 0.
        """
        df = create_target(raw_sample_df)

        # G3 values were: [12, 8, 15, 5, 16, 11, 9, 10]
        # Expected labels:  [ 1, 0,  1, 0,  1,  1, 0,  1]
        expected = [1, 0, 1, 0, 1, 1, 0, 1]
        assert list(df[TARGET_COL]) == expected, (
            f"pass_fail values are wrong.\n"
            f"Expected: {expected}\n"
            f"Got:      {list(df[TARGET_COL])}"
        )

    def test_boundary_exactly_10_is_pass(self, raw_sample_df):
        """
        G3 == 10 is the pass/fail boundary — it should be a PASS (= 1).
        """
        df = create_target(raw_sample_df)
        # Row index 7 has G3 = 10 → should be 1
        assert df.iloc[7][TARGET_COL] == 1, "G3 == 10 should be pass (1)"

    def test_grade_columns_are_dropped(self, raw_sample_df):
        """
        G1, G2, G3 must NOT appear as features after create_target().
        """
        df = create_target(raw_sample_df)
        for col in ["G1", "G2", "G3"]:
            assert col not in df.columns, f"Column '{col}' should have been dropped"

    def test_target_column_is_binary(self, raw_sample_df):
        """
        pass_fail should contain only 0s and 1s — nothing else.
        """
        df = create_target(raw_sample_df)
        unique_values = set(df[TARGET_COL].unique())
        assert unique_values.issubset({0, 1}), (
            f"Target column has unexpected values: {unique_values}"
        )

    def test_row_count_unchanged(self, raw_sample_df):
        """
        create_target() should not drop any rows — only add a column and drop 3.
        """
        df = create_target(raw_sample_df)
        assert len(df) == len(raw_sample_df), "Row count changed unexpectedly"


# ------------------------------------------------------------------
# TEST 2: select_features()
# ------------------------------------------------------------------

class TestSelectFeatures:

    def test_returns_only_expected_columns(self, raw_sample_df):
        """
        select_features() must return exactly FEATURE_COLS + TARGET_COL.
        Any extra columns (school, famsize, etc.) must be dropped.
        """
        df = create_target(raw_sample_df)
        df = select_features(df)

        expected_cols = set(FEATURE_COLS + [TARGET_COL])
        actual_cols   = set(df.columns)

        assert actual_cols == expected_cols, (
            f"Column mismatch.\n"
            f"Extra (should not be there): {actual_cols - expected_cols}\n"
            f"Missing (should be there):   {expected_cols - actual_cols}"
        )

    def test_column_count_is_correct(self, raw_sample_df):
        """
        We expect exactly 12 feature columns + 1 target = 13 total.
        """
        df = create_target(raw_sample_df)
        df = select_features(df)
        assert df.shape[1] == 13, (
            f"Expected 13 columns (12 features + target), got {df.shape[1]}"
        )

    def test_no_rows_dropped(self, raw_sample_df):
        """
        Column selection should never change the number of rows.
        """
        df = create_target(raw_sample_df)
        df = select_features(df)
        assert len(df) == len(raw_sample_df)


# ------------------------------------------------------------------
# TEST 3: split_data()
# ------------------------------------------------------------------

class TestSplitData:

    def test_train_test_proportion_approximate(self, raw_sample_df):
        """
        With test_size=0.2, ~80% of rows should be in train and ~20% in test.
        We allow a small tolerance because stratification on tiny datasets
        can shift proportions slightly.
        """
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)
        X_train, X_test, y_train, y_test = split_data(df)

        total = len(X_train) + len(X_test)
        assert total == len(df), "Train + test sizes don't add up to total"

        # Allow ±1 row tolerance for small datasets
        expected_test  = round(len(df) * 0.2)
        assert abs(len(X_test) - expected_test) <= 1, (
            f"Test set should have ~{expected_test} rows, got {len(X_test)}"
        )

    def test_no_overlap_between_train_and_test(self, raw_sample_df):
        """
        Critical: same row must NOT appear in both train and test.
        """
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)
        X_train, X_test, y_train, y_test = split_data(df)

        # Index values must be disjoint
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        overlap   = train_idx & test_idx
        assert len(overlap) == 0, f"Found {len(overlap)} rows in both train and test!"

    def test_stratification_preserves_class_ratio(self, raw_sample_df):
        """
        The pass rate in train and test should be close to the overall pass rate.
        Tolerance: ±15% (wider because our fixture only has 8 rows).
        """
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)
        X_train, X_test, y_train, y_test = split_data(df)

        overall_pass_rate = (raw_sample_df["G3"] >= 10).mean()
        train_pass_rate   = y_train.mean()

        assert abs(train_pass_rate - overall_pass_rate) <= 0.15, (
            f"Train pass rate ({train_pass_rate:.2%}) deviates too much "
            f"from overall ({overall_pass_rate:.2%})"
        )

    def test_reproducibility_same_split_every_time(self, raw_sample_df):
        """
        Two calls with the same random_state must produce identical splits.
        """
        df = create_target(raw_sample_df)
        df = select_features(df)
        df = encode_features(df)

        X_train_1, X_test_1, _, _ = split_data(df, random_state=42)
        X_train_2, X_test_2, _, _ = split_data(df, random_state=42)

        assert list(X_train_1.index) == list(X_train_2.index), (
            "Same random_state should give same train set every time"
        )
