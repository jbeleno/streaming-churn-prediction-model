"""Preprocessing utilities for the streaming churn dataset.

The pipeline:
1. Engineers tenure features (`customer_tenure_days`, `customer_tenure_years`)
   from `signup_date` against a fixed BASE_DATE.
2. Drops the raw `signup_date` column.
3. Applies a ColumnTransformer that:
   - Standard-scales numerical columns.
   - One-hot encodes categorical columns (with handle_unknown='ignore').

Note: the original notebook trained tree-based models on label-encoded data
and the logistic regression on one-hot data. We unify on **one-hot** because:
- Modern scikit-learn pipelines work better with consistent transforms.
- RF/XGBoost handle wide one-hot inputs without performance penalty on this
  dataset size (~125k × ~30 features after OHE).
- A single Pipeline simplifies persistence: train and serve with the same
  preprocessor.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BASE_DATE,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    SIGNUP_DATE_COL,
    TARGET_COL,
    TENURE_DAYS_COL,
    TENURE_YEARS_COL,
)


def add_tenure_features(df: pd.DataFrame, base_date: str = BASE_DATE) -> pd.DataFrame:
    """Add ``customer_tenure_days`` and ``customer_tenure_years`` columns.

    Drops ``signup_date`` after computing the derived features.
    """
    df = df.copy()
    base = pd.to_datetime(base_date)
    signup = pd.to_datetime(df[SIGNUP_DATE_COL])

    df[TENURE_DAYS_COL] = (base - signup).dt.days
    df[TENURE_YEARS_COL] = df[TENURE_DAYS_COL] / 365.0

    return df.drop(columns=[SIGNUP_DATE_COL])


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used inside every model pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) split. Assumes ``add_tenure_features`` already ran."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataframe")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def make_pipeline(model) -> Pipeline:
    """Wrap the preprocessor + an estimator in a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
