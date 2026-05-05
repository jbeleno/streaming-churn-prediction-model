"""Configuration constants for the streaming churn pipeline.

Single source of truth for column names, base dates, hyperparameter grids
and metric definitions. Keeping these here (instead of inline in train.py
or scattered across a notebook) makes the pipeline reproducible and easy
to tweak from one place.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

TRAIN_CSV: Path = DATA_DIR / "train.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"

# --- Columns -----------------------------------------------------------------

TARGET_COL: str = "churned"
SIGNUP_DATE_COL: str = "signup_date"
TENURE_DAYS_COL: str = "customer_tenure_days"
TENURE_YEARS_COL: str = "customer_tenure_years"

CATEGORICAL_COLS: list[str] = [
    "location",
    "subscription_type",
    "payment_plan",
    "payment_method",
    "customer_service_inquiries",
]

# Numerical features used by all models.
# ``customer_tenure_days`` and ``customer_tenure_years`` are engineered features
# computed in ``preprocessing.add_tenure_features``.
NUMERIC_COLS: list[str] = [
    "age",
    "weekly_hours",
    "average_session_length",
    "song_skip_rate",
    "weekly_songs_played",
    "weekly_unique_songs",
    "num_subscription_pauses",
    "num_favorite_artists",
    "num_platform_friends",
    "num_playlists_created",
    "num_shared_playlists",
    "notifications_clicked",
    TENURE_DAYS_COL,
    TENURE_YEARS_COL,
]

# Reference "today" used to compute customer_tenure_*. Frozen for reproducibility.
BASE_DATE: str = "2025-07-16"

# --- Reproducibility ---------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_SPLITS: int = 5

# --- Hyperparameter grids ----------------------------------------------------
# Kept identical to the original notebook for direct comparability.

RF_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

XGB_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [100, 300],
    "model__learning_rate": [0.01, 0.1],
    "model__max_depth": [3, 6],
}

LOGREG_PARAM_GRID: dict[str, list] = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l2"],
    "model__solver": ["liblinear", "lbfgs"],
}

# Default search budget; override with --n-iter on the CLI for quick smoke tests.
DEFAULT_N_ITER: int = 20
