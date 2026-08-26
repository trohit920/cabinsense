"""Single-head image classifier (drowsiness / eye-state).

A backbone plus a dropout-regularized linear head. Used for the binary drowsiness
model in Phase 2; the same class serves any single-label classification task.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import BACKBONE
from src.models.backbones import build_backbone


class ImageClassifier(nn.Module):
    """Backbone + linear classification head."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Build a classifier with the given backbone and number of output classes."""
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, num_classes)."""
        return self.head(self.backbone(images))
