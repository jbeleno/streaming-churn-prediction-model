"""Smoke tests for the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BASE_DATE,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    TENURE_DAYS_COL,
    TENURE_YEARS_COL,
)
from src.preprocessing import (
    add_tenure_features,
    build_preprocessor,
    make_pipeline,
    split_features_target,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal valid dataframe with one row per category."""
    return pd.DataFrame(
        {
            "signup_date": ["2024-01-15", "2023-06-01", "2025-03-20"],
            "age": [25, 35, 45],
            "weekly_hours": [10.5, 8.0, 12.0],
            "average_session_length": [1.2, 0.9, 1.5],
            "song_skip_rate": [0.1, 0.3, 0.2],
            "weekly_songs_played": [50, 30, 70],
            "weekly_unique_songs": [40, 25, 60],
            "num_subscription_pauses": [0, 1, 0],
            "num_favorite_artists": [10, 5, 15],
            "num_platform_friends": [3, 1, 5],
            "num_playlists_created": [2, 0, 4],
            "num_shared_playlists": [1, 0, 2],
            "notifications_clicked": [5, 2, 8],
            "location": ["US", "MX", "ES"],
            "subscription_type": ["Premium", "Free", "Family"],
            "payment_plan": ["Monthly", "Yearly", "Monthly"],
            "payment_method": ["Card", "PayPal", "Card"],
            "customer_service_inquiries": ["None", "Some", "Many"],
            TARGET_COL: [0, 1, 0],
        }
    )


def test_add_tenure_features_creates_two_columns(sample_df):
    out = add_tenure_features(sample_df, base_date=BASE_DATE)

    assert TENURE_DAYS_COL in out.columns
    assert TENURE_YEARS_COL in out.columns
    assert "signup_date" not in out.columns


def test_add_tenure_features_yields_positive_days(sample_df):
    out = add_tenure_features(sample_df, base_date=BASE_DATE)

    assert (out[TENURE_DAYS_COL] > 0).all()
    np.testing.assert_allclose(
        out[TENURE_YEARS_COL], out[TENURE_DAYS_COL] / 365.0, rtol=1e-6
    )


def test_split_features_target_drops_target(sample_df):
    df = add_tenure_features(sample_df)
    X, y = split_features_target(df)

    assert TARGET_COL not in X.columns
    assert len(X) == len(y) == 3
    assert y.tolist() == [0, 1, 0]


def test_split_features_target_raises_when_missing(sample_df):
    df = add_tenure_features(sample_df).drop(columns=[TARGET_COL])
    with pytest.raises(ValueError, match=TARGET_COL):
        split_features_target(df)


def test_preprocessor_handles_full_dataframe(sample_df):
    df = add_tenure_features(sample_df)
    X, _ = split_features_target(df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    # Numeric features remain (count) + one-hot adds (n_cat - 1) per categorical
    expected_min_cols = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)  # at least 1 per cat
    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= expected_min_cols


def test_preprocessor_handles_unknown_category(sample_df):
    """OneHotEncoder must not fail on unseen categories at inference time."""
    df = add_tenure_features(sample_df)
    X, _ = split_features_target(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    # Inject an unseen category in 'location'.
    X_unknown = X.copy()
    X_unknown.loc[0, "location"] = "ZZ_UNKNOWN"
    transformed = preprocessor.transform(X_unknown)

    assert transformed.shape[0] == 3  # no rows dropped
    # The unknown category encodes as all zeros for that one-hot block — that's fine.


def test_make_pipeline_wraps_estimator(sample_df):
    """A pipeline with a dummy estimator should fit + predict end-to-end."""
    from sklearn.linear_model import LogisticRegression

    df = add_tenure_features(sample_df)
    X, y = split_features_target(df)

    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == 3
    assert set(preds).issubset({0, 1})
