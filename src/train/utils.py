"""Training utilities: deterministic seeding.

Centralizes seeding of Python, numpy, and torch (CPU + CUDA) so training and
evaluation runs are reproducible, per the project's SEED constant.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, numpy, and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
