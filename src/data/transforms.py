"""Image augmentation and preprocessing transforms (torchvision v2).

Training transforms inject the vehicle-condition variations that matter in a cabin -
lighting (photometric jitter), occlusion (random erasing), and pose (small affine)
- while eval transforms are deterministic. albumentations is intentionally not used
(its stringzilla dependency fails to build on this Windows toolchain); torchvision
v2 covers the same augmentations.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from src.config import IMAGENET_MEAN, IMAGENET_STD


def build_train_transforms(size: int, allow_flip: bool = True) -> v2.Compose:
    """Training transforms with lighting/occlusion/pose augmentation.

    ``allow_flip`` should be False for tasks where left/right matters (e.g. gaze);
    it is safe for symmetric attribute tasks (drowsiness, glasses, cap).
    """
    steps: list[v2.Transform] = [v2.Resize((size, size), antialias=True)]
    if allow_flip:
        steps.append(v2.RandomHorizontalFlip(p=0.5))
    steps.extend(
        [
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),  # lighting
            v2.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.1)),  # pose
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # occlusion
        ]
    )
    return v2.Compose(steps)


def build_eval_transforms(size: int) -> v2.Compose:
    """Deterministic transforms for validation/test/inference."""
    return v2.Compose(
        [
            v2.Resize((size, size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
