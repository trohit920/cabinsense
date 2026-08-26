"""Unit tests for preprocessing, landmark selection, and manifest validation."""

from __future__ import annotations

import numpy as np

from src.core.landmark_indices import (
    left_eye_points,
    mouth_points,
    pose_points,
    right_eye_points,
)
from src.core.manifest import validate_manifest
from src.core.preprocess import preprocess_crop


def test_preprocess_crop_shape_and_dtype() -> None:
    """A BGR crop becomes a (1, 3, size, size) float32 tensor."""
    image = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)
    out = preprocess_crop(image, size=64)
    assert out.shape == (1, 3, 64, 64)
    assert out.dtype == np.float32


def test_landmark_selection_shapes() -> None:
    """Landmark selectors return the expected point counts from a 98-point set."""
    landmarks = np.random.rand(98, 2).astype(np.float32)
    assert left_eye_points(landmarks).shape == (6, 2)
    assert right_eye_points(landmarks).shape == (6, 2)
    assert mouth_points(landmarks).shape == (6, 2)
    assert pose_points(landmarks).shape == (5, 2)


def test_manifest_validation_detects_errors() -> None:
    """A valid manifest passes; a malformed one reports specific errors."""
    good = {
        "models": {
            "drowsiness": {
                "file": "drowsiness.onnx",
                "input_name": "input",
                "input_shape": [1, 3, 128, 128],
                "layout": "NCHW",
                "color": "RGB",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "output_names": ["logits"],
            }
        }
    }
    assert validate_manifest(good) == []
    assert validate_manifest({"models": {}})  # empty -> error(s)
    bad = {"models": {"m": {"file": "m.onnx", "input_shape": [1, 3]}}}
    errors = validate_manifest(bad)
    assert any("input_shape" in e for e in errors)
    assert any("missing required key" in e for e in errors)
