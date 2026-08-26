"""Evaluation metrics (accuracy/F1, masked multi-task, latency, stability).

Pure functions over predictions and gold labels: single-label classification
metrics, masked per-task metrics for the partial-label multi-task model, latency
statistics, and a temporal-stability (flicker) metric. Deterministic and
unit-testable; the accuracy / latency / stability triad computed locally.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Accuracy and macro-F1 for single-label classification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def masked_multitask_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, tasks: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-task accuracy/F1 over only the labeled (masked) samples for each task."""
    results: dict[str, dict[str, float]] = {}
    for index, task in enumerate(tasks):
        valid = mask[:, index] > 0
        support = int(valid.sum())
        if support == 0:
            results[task] = {"accuracy": float("nan"), "f1": float("nan"), "support": 0}
            continue
        true_col = y_true[valid, index].astype(int)
        pred_col = y_pred[valid, index].astype(int)
        results[task] = {
            "accuracy": float(accuracy_score(true_col, pred_col)),
            "f1": float(f1_score(true_col, pred_col, zero_division=0)),
            "support": support,
        }
    return results


def latency_stats(times_ms: Sequence[float]) -> dict[str, float]:
    """Median/p95/mean latency (ms) and derived FPS from per-inference timings."""
    if not times_ms:
        return {}
    array = np.asarray(times_ms, dtype=np.float64)
    median = float(np.median(array))
    return {
        "median_ms": median,
        "p95_ms": float(np.percentile(array, 95)),
        "mean_ms": float(array.mean()),
        "fps": float(1000.0 / median) if median > 0 else 0.0,
    }


def flicker_rate(sequence: Sequence[str]) -> float:
    """Fraction of adjacent frames whose label changed (lower = more stable)."""
    if len(sequence) < 2:
        return 0.0
    changes = sum(1 for a, b in itertools.pairwise(sequence) if a != b)
    return changes / (len(sequence) - 1)
