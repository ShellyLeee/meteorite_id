"""Classification metrics."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Compute accuracy."""
    return float(sk_accuracy_score(y_true, y_pred))


def precision(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary precision with safe zero handling."""
    return float(sk_precision_score(y_true, y_pred, average="binary", zero_division=0))


def recall(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary recall with safe zero handling."""
    return float(sk_recall_score(y_true, y_pred, average="binary", zero_division=0))


def f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary F1 score with safe zero handling."""
    return float(sk_f1_score(y_true, y_pred, average="binary", zero_division=0))


def tune_threshold(
    y_true: list[int],
    y_prob: list[float],
    thresholds: list[float] | None = None,
) -> tuple[float, float]:
    """Tune a binary threshold by maximizing validation F1."""
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not y_true:
        return 0.5, 0.0

    if thresholds is None:
        thresholds = [i / 100 for i in range(20, 81)]

    best_threshold = 0.5
    best_f1 = float("-inf")

    for threshold in thresholds:
        preds = [1 if prob >= threshold else 0 for prob in y_prob]
        score = f1(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Compute the full classification metric set."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
    }
