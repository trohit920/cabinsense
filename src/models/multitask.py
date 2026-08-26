"""Multi-task attribute model (shared backbone + per-task binary heads).

One shared feature extractor feeds an independent binary head per attribute task
(glasses, cap, seatbelt, phone, smoking, yawn). Trained with a masked BCE loss so
each sample supervises only the tasks its source dataset labels - supporting the
partial-label, multi-dataset setting. Emits stacked logits aligned with the task
tuple for easy masking.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ATTRIBUTE_TASKS, BACKBONE
from src.models.backbones import build_backbone


class MultiTaskClassifier(nn.Module):
    """Shared backbone with one binary head per attribute task."""

    def __init__(
        self,
        tasks: tuple[str, ...] = ATTRIBUTE_TASKS,
        backbone_name: str = BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Build the shared backbone and one linear head per task."""
        super().__init__()
        self.tasks = tasks
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in tasks])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-task logits of shape (batch, num_tasks), ordered like ``tasks``."""
        features = self.dropout(self.backbone(images))
        return torch.cat([head(features) for head in self.heads], dim=1)
