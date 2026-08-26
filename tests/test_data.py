"""Unit tests for augmentation transforms and the seeded split (no downloads)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.datasets import seeded_split
from src.data.transforms import build_eval_transforms, build_train_transforms


def test_eval_transform_output_shape() -> None:
    """Eval transform yields a (3, size, size) float tensor from a PIL image."""
    image = Image.fromarray(np.random.randint(0, 255, (200, 180, 3), dtype=np.uint8))
    tensor = build_eval_transforms(128)(image)
    assert tuple(tensor.shape) == (3, 128, 128)
    assert tensor.dtype.is_floating_point


def test_train_transform_output_shape() -> None:
    """Train transform (with augmentation) yields the same output shape."""
    image = Image.fromarray(np.random.randint(0, 255, (200, 180, 3), dtype=np.uint8))
    tensor = build_train_transforms(96)(image)
    assert tuple(tensor.shape) == (3, 96, 96)


def test_seeded_split_is_deterministic_and_partitions() -> None:
    """The split is reproducible and forms a disjoint cover of all indices."""
    a = seeded_split(100)
    b = seeded_split(100)
    for key in ("train", "val", "test"):
        assert np.array_equal(a[key], b[key])
    combined = np.concatenate([a["train"], a["val"], a["test"]])
    assert sorted(combined.tolist()) == list(range(100))


def test_split_ratios_must_sum_to_one() -> None:
    """Invalid ratios are rejected."""
    with pytest.raises(ValueError):
        seeded_split(100, ratios=(0.5, 0.3, 0.1))
