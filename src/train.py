"""Training entrypoint for the streaming churn pipeline.

Usage:
    python -m src.train                       # full search (n_iter=20)
    python -m src.train --n-iter 3 --quick    # smoke test (~1 min)
    python -m src.train --models rf logreg    # subset of models
    python -m src.train --output ./artifacts  # custom artifact dir

Artifacts written to ``artifacts/``:
    - <model>.joblib       — pipeline (preprocessor + model)
    - metrics.json         — validation metrics per model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    CV_SPLITS,
    DEFAULT_N_ITER,
    LOGREG_PARAM_GRID,
    RANDOM_STATE,
    RF_PARAM_GRID,
    TEST_SIZE,
    TRAIN_CSV,
    XGB_PARAM_GRID,
)
from src.eval import evaluate, metrics_table
from src.preprocessing import add_tenure_features, make_pipeline, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

ALL_MODELS = ("rf", "xgboost", "logreg")


def _make_search(model_key: str, n_iter: int, quick: bool):
    """Return (estimator, search_strategy, param_grid) for a given model key."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(roc_auc_score, response_method="predict_proba")

    if model_key == "rf":
        pipeline = make_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        )
        return RandomizedSearchCV(
            pipeline,
            param_distributions=RF_PARAM_GRID,
            n_iter=min(n_iter, _grid_size(RF_PARAM_GRID)),
            scoring=scorer,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )

    if model_key == "xgboost":
        pipeline = make_pipeline(
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )
        )
        return RandomizedSearchCV(
            pipeline,
            param_distributions=XGB_PARAM_GRID,
            n_iter=min(n_iter, _grid_size(XGB_PARAM_GRID)),
            scoring=scorer,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )

    if model_key == "logreg":
        pipeline = make_pipeline(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        )
        # Grid for logreg is small — exhaustive search.
        if quick:
            return GridSearchCV(
                pipeline,
                param_grid={"model__C": [1.0], "model__penalty": ["l2"], "model__solver": ["lbfgs"]},
                scoring=scorer,
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
        return GridSearchCV(
            pipeline,
            param_grid=LOGREG_PARAM_GRID,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

    raise ValueError(f"Unknown model: {model_key}")


def _grid_size(grid: dict) -> int:
    """Number of combinations in a param grid."""
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _load_data(quick: bool, train_csv: Path) -> pd.DataFrame:
    log.info("Loading training data from %s", train_csv)
    df = pd.read_csv(train_csv)
    if quick:
        # Stratified subsample to keep class balance.
        df = df.sample(n=min(5000, len(df)), random_state=RANDOM_STATE)
        log.info("Quick mode: subsampled to %d rows", len(df))
    df = add_tenure_features(df)
    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_MODELS),
        choices=ALL_MODELS,
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help="RandomizedSearchCV iterations (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Subsample to 5k rows and skip exhaustive logreg grid (fast smoke test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Where to write trained models and metrics.json",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=TRAIN_CSV,
        help="Path to training CSV",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    df = _load_data(quick=args.quick, train_csv=args.train_csv)
    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info("Train: %d rows · Val: %d rows", len(X_train), len(X_val))

    results = []
    for model_key in args.models:
        log.info("--- Training %s ---", model_key)
        search = _make_search(model_key, n_iter=args.n_iter, quick=args.quick)
        search.fit(X_train, y_train)

        best = search.best_estimator_
        log.info("Best params for %s: %s", model_key, search.best_params_)
        log.info("Best CV AUC for %s: %.4f", model_key, search.best_score_)

        metrics = evaluate(best, X_val, y_val, name=model_key)
        print(metrics)
        results.append(metrics)

        artifact_path = args.output / f"{model_key}.joblib"
        joblib.dump(best, artifact_path)
        log.info("Saved %s to %s", model_key, artifact_path)

    # Persist metrics summary.
    summary = metrics_table(results).to_dict(orient="records")
    metrics_path = args.output / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote metrics summary to %s", metrics_path)

    print("\n=== Final ranking ===")
    print(metrics_table(results).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
