"""Evaluation helpers: metrics and (optional) plots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for a binary classifier on a held-out validation set."""

    name: str
    auc_roc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray

    def as_row(self) -> dict[str, float | str]:
        """Flat dict suitable for pandas.DataFrame construction."""
        return {
            "model": self.name,
            "auc_roc": round(self.auc_roc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }

    def __str__(self) -> str:
        return (
            f"=== {self.name} ===\n"
            f"AUC-ROC:   {self.auc_roc:.4f}\n"
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1-score:  {self.f1:.4f}\n"
            f"Confusion Matrix:\n{self.confusion}"
        )


def evaluate(model, X_val, y_val, name: str) -> ModelMetrics:
    """Compute the standard set of binary-classification metrics."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return ModelMetrics(
        name=name,
        auc_roc=roc_auc_score(y_val, y_proba),
        accuracy=accuracy_score(y_val, y_pred),
        precision=precision_score(y_val, y_pred),
        recall=recall_score(y_val, y_pred),
        f1=f1_score(y_val, y_pred),
        confusion=confusion_matrix(y_val, y_pred),
    )


def metrics_table(results: list[ModelMetrics]) -> pd.DataFrame:
    """Return a pandas DataFrame ordered by AUC-ROC desc."""
    df = pd.DataFrame([m.as_row() for m in results])
    return df.sort_values(by="auc_roc", ascending=False).reset_index(drop=True)
