from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    
    """
    Here we compute standart binary classification metrics.

    Returns a dictionary with accuracy, precision, recall, binary F1 and macro F1.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_metrics_row(
        model_name: str,
        split_name: str,
        y_true,
        y_pred,
) -> dict[str, Any]:
    
    """
    Building one flat metrics row (in case of CSV export).
    """

    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "model_name": model_name,
        "split": split_name,
        "n_examples": int(len(y_true)),
        **metrics,
    }

