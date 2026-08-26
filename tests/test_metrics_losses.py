"""Unit tests for evaluation metrics and the masked multi-task loss."""

from __future__ import annotations

import numpy as np
import torch

from src.eval.metrics import (
    classification_metrics,
    flicker_rate,
    latency_stats,
    masked_multitask_metrics,
)
from src.train.losses import masked_bce_with_logits


def test_classification_metrics_perfect_and_wrong() -> None:
    """Accuracy is 1.0 for perfect predictions and lower otherwise."""
    assert classification_metrics([0, 1, 1], [0, 1, 1])["accuracy"] == 1.0
    assert classification_metrics([0, 1], [1, 0])["accuracy"] == 0.0


def test_masked_multitask_metrics_uses_only_labeled() -> None:
    """Only masked-in entries count; masked-out tasks report zero support."""
    y_true = np.array([[1, 0], [0, 0]], dtype=np.float32)
    y_pred = np.array([[1, 1], [0, 1]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 0]], dtype=np.float32)  # only task 0 labeled
    result = masked_multitask_metrics(y_true, y_pred, mask, ("glasses", "cap"))
    assert result["glasses"]["support"] == 2
    assert result["glasses"]["accuracy"] == 1.0
    assert result["cap"]["support"] == 0


def test_latency_and_flicker() -> None:
    """Latency stats derive FPS from the median; flicker counts label changes."""
    stats = latency_stats([10.0, 20.0, 30.0])
    assert stats["median_ms"] == 20.0
    assert abs(stats["fps"] - 50.0) < 1e-6
    assert flicker_rate(["road", "road", "left", "left"]) == 1 / 3
    assert flicker_rate(["road"]) == 0.0


def test_masked_bce_ignores_masked_entries() -> None:
    """Fully-masked batches give zero loss; confident-correct preds give small loss."""
    logits = torch.tensor([[5.0, 5.0]])
    targets = torch.tensor([[1.0, 1.0]])
    zero_mask = torch.zeros_like(targets)
    assert float(masked_bce_with_logits(logits, targets, zero_mask)) == 0.0
    full_mask = torch.ones_like(targets)
    assert float(masked_bce_with_logits(logits, targets, full_mask)) < 0.05
