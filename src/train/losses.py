"""Loss functions for training.

Provides the masked binary-cross-entropy used by the multi-task attribute model, so
each sample only contributes to the tasks its source dataset labels (partial-label
learning). Pure tensor logic, unit-testable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean BCE over labeled (mask==1) task entries only.

    Shapes are all (batch, num_tasks). Entries with mask 0 are excluded; if no
    entries are labeled the loss is zero (a safe no-op for that batch).
    """
    per_element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = per_element * mask
    denominator = mask.sum().clamp_min(1.0)
    return masked.sum() / denominator
