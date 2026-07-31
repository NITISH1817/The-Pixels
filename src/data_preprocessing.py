"""
src/data_preprocessing.py
--------------------------
Reusable data preprocessing functions for the Student Performance MLOps project.

Each function does ONE thing clearly so they can be tested, reused, and swapped
individually without breaking the rest of the pipeline.

Usage (from project root, with .venv activated):
    from src.data_preprocessing import load_data, create_target, select_features, \
                                        encode_features, split_data
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Which columns are categorical (need encoding) vs numeric (keep as-is)
# ------------------------------------------------------------------

# Categorical features (will be one-hot encoded)
CATEGORICAL_COLS = [
    "sex",      # Male / Female
    "address",  # Urban / Rural
    "Mjob",     # Mother's job
    "Fjob",     # Father's job
    "famsup",   # Family educational support
    "internet", # Internet access at home
    "higher",   # Wants to pursue higher education -- STRONG predictor!
    "schoolsup",# Extra school support
    "paid",     # Paid extra classes
    "romantic", # In a romantic relationship
    "Pstatus",  # Parents living together (T) or apart (A)
]

# Numeric features (kept as numbers, scaled during training)
NUMERIC_COLS = [
    "age",
    "studytime",  # Weekly study time (1-4)
    "failures",   # Past class failures
    "Medu",       # Mother's education (0-4)
    "Fedu",       # Father's education (0-4)
    "absences",   # Number of absences
    "famrel",     # Family relationship quality (1-5)
    "freetime",   # Free time after school (1-5)
    "goout",      # Going out with friends (1-5) -- more = less studying
    "Walc",       # Weekend alcohol consumption (1-5)
    "Dalc",       # Workday alcohol consumption (1-5)
    "health",     # Current health status (1-5)
    "traveltime", # Home-to-school travel time (1-4)
]

# All input features we actually use (everything else gets dropped)
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# The target column we engineer from G3
TARGET_COL = "pass_fail"



# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load the student-mat.csv file.

    The file uses semicolons (;) as separators, NOT commas.
    pandas automatically strips surrounding quotes from string values.

    Args:
        path: Relative or absolute path to the CSV file.

    Returns:
        Raw DataFrame with all original 33 columns.
    """
    df = pd.read_csv(path, sep=";")

    # G1 and G2 are stored as quoted strings ("5", "6") in this dataset.
    # Convert them to integers so we can use them safely if ever needed.
    for col in ["G1", "G2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[load_data] Loaded {len(df)} rows, {len(df.columns)} columns from '{path}'")
    return df


# ------------------------------------------------------------------
# 2. CREATE TARGET
# ------------------------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the binary target column 'pass_fail' from G3 (final grade).

    Rule:
        pass_fail = 1  if G3 >= 10  (Portuguese grading: 10/20 is the pass mark)
        pass_fail = 0  if G3 <  10

    WHY drop G1 and G2?
        We are predicting student performance EARLY — before any exam results
        exist. Using G1/G2 would be "data leakage" (using future information
        to predict the future), which makes the model useless in practice.

    Args:
        df: Raw DataFrame (output of load_data).

    Returns:
        DataFrame with 'pass_fail' added and G1, G2, G3 removed.
    """
    df = df.copy()  # never modify the original DataFrame in-place

    # Create the binary label
    df[TARGET_COL] = (df["G3"] >= 10).astype(int)

    # Count how many passed vs failed (useful sanity check)
    pass_count = df[TARGET_COL].sum()
    fail_count = len(df) - pass_count
    print(f"[create_target] Pass: {pass_count} ({pass_count/len(df)*100:.1f}%)  "
          f"Fail: {fail_count} ({fail_count/len(df)*100:.1f}%)")

    # Drop grade columns — they must NOT be used as model inputs
    df = df.drop(columns=["G1", "G2", "G3"])

    return df


# ------------------------------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------------------------------
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the input features we want plus the target column.

    WHY these 12 features?
        They represent information available at the START of the school year
        (demographics, family background, study habits) — not end-of-year data.

    Features kept:
        Numeric:     age, studytime, failures, Medu, Fedu, absences
        Categorical: sex, address, Mjob, Fjob, famsup, internet

    Args:
        df: DataFrame after create_target() has been applied.

    Returns:
        DataFrame with only FEATURE_COLS + TARGET_COL columns.
    """
    cols_to_keep = FEATURE_COLS + [TARGET_COL]

    # Safety check: warn if any expected column is missing
    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        raise ValueError(f"[select_features] Missing columns in DataFrame: {missing}")

    df = df[cols_to_keep].copy()
    print(f"[select_features] Kept {len(FEATURE_COLS)} features + target. "
          f"Shape: {df.shape}")
    return df


# ------------------------------------------------------------------
# 4. ENCODE FEATURES
# ------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns; leave numeric columns unchanged.

    WHY one-hot encoding?
        Machine learning models work with numbers, not strings like "yes"/"no"
        or "at_home"/"teacher". One-hot encoding converts each category into
        a separate 0/1 column — so "Mjob_teacher", "Mjob_at_home", etc.

    We use drop_first=True to avoid the "dummy variable trap" (multicollinearity):
        If we have columns for every category, any one column is perfectly
        predictable from the others — which confuses models like Logistic Regression.

    Args:
        df: DataFrame after select_features() has been applied.

    Returns:
        DataFrame with categorical columns replaced by one-hot columns.
        Target column (pass_fail) is kept as-is.
    """
    df = df.copy()

    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True,   # avoids dummy variable trap
        dtype=int          # produce 0/1 integers instead of booleans
    )

    print(f"[encode_features] After encoding: {df_encoded.shape[1]} columns "
          f"(was {df.shape[1]})")
    return df_encoded


# ------------------------------------------------------------------
# 5. SPLIT DATA
# ------------------------------------------------------------------
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train and test sets (80% / 20% by default).

    WHY stratify?
        If pass/fail is imbalanced (e.g., 70% pass), a random split might put
        all fails in training and none in test. Stratification ensures both sets
        have the same pass/fail ratio as the full dataset.

    WHY random_state=42?
        Reproducibility. Anyone who runs this code gets the exact same split,
        so results are comparable and the project is reproducible.

    Args:
        df:           Encoded DataFrame (output of encode_features).
        test_size:    Fraction of data for the test set (default 0.2 = 20%).
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test  (four DataFrames/Series)
    """
    # Separate features (X) from target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,          # maintain class balance in both splits
        random_state=random_state
    )

    print(f"[split_data] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"[split_data] Train pass rate: {y_train.mean():.2%} | "
          f"Test pass rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# 6. CONVENIENCE: run the full pipeline in one call
# ------------------------------------------------------------------
def run_full_pipeline(data_path: str):
    """
    Run all preprocessing steps in order and return the final split.

    This is a convenience wrapper — the individual functions still exist
    so they can be called separately (e.g., in tests or the API).

    Args:
        data_path: Path to the raw student-mat.csv file.

    Returns:
        X_train, X_test, y_train, y_test
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
