"""Backbone factory.

Builds a pooled feature-extractor backbone from timm (MobileNetV3-Small by default,
MobileViT for the CNN-vs-Transformer ablation) and reports its true feature
dimension via a dummy forward. Edge-appropriate backbones keep the on-device
latency/VRAM budget realistic.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.config import BACKBONE


def build_backbone(name: str = BACKBONE, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return a global-pooled backbone and its true output feature dimension.

    The dimension is measured with a dummy forward rather than read from
    ``num_features``, which for some backbones (e.g. MobileNetV3) reports the block
    channels (576) instead of the post-conv-head classifier input (1024).
    """
    backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    with torch.no_grad():
        feature_dim = int(backbone(torch.zeros(1, 3, 96, 96)).shape[1])
    return backbone, feature_dim
