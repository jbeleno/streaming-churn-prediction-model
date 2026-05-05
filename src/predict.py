"""Inference entrypoint.

Usage:
    python -m src.predict --model artifacts/rf.joblib --input test.csv --output predictions.csv

Reads a CSV that has the same schema as ``train.csv`` minus the ``churned``
column. Applies the saved Pipeline (preprocessor + estimator) and writes
predictions + churn probabilities back to disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.config import TARGET_COL
from src.preprocessing import add_tenure_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a .joblib file produced by src.train",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with the input rows (same schema as train.csv minus 'churned')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Where to write the predictions CSV",
    )
    args = parser.parse_args()

    log.info("Loading model from %s", args.model)
    pipeline = joblib.load(args.model)

    log.info("Loading input from %s", args.input)
    df = pd.read_csv(args.input)
    df = add_tenure_features(df)

    # Drop the target column if it is present (e.g. when running on labelled data).
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    log.info("Predicting %d rows...", len(df))
    y_pred = pipeline.predict(df)
    y_proba = pipeline.predict_proba(df)[:, 1]

    out = pd.DataFrame({"prediction": y_pred, "churn_probability": y_proba.round(6)})
    out.to_csv(args.output, index=False)
    log.info("Wrote %d predictions to %s", len(out), args.output)


if __name__ == "__main__":
    sys.exit(main())
