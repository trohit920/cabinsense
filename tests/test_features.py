"""Unit tests for EAR/MAR/PERCLOS geometry features."""

from __future__ import annotations

import numpy as np

from src.core.features import eye_aspect_ratio, mouth_aspect_ratio, perclos

# Order: [left_corner, top1, top2, right_corner, bottom2, bottom1].
_OPEN_EYE = np.array([[0, 1], [1, 0], [3, 0], [4, 1], [3, 2], [1, 2]], dtype=np.float32)
_CLOSED_EYE = np.array([[0, 1], [1, 0.9], [3, 0.9], [4, 1], [3, 1.1], [1, 1.1]], dtype=np.float32)


def test_ear_high_when_open_low_when_closed() -> None:
    """EAR is large for an open eye and small for a closed eye."""
    assert eye_aspect_ratio(_OPEN_EYE) > 0.3
    assert eye_aspect_ratio(_CLOSED_EYE) < 0.1


def test_mar_high_when_mouth_open() -> None:
    """MAR grows as the mouth opens (yawn)."""
    closed_mouth = np.array([[0, 1], [1, 0.9], [3, 0.9], [4, 1], [3, 1.1], [1, 1.1]], dtype=np.float32)
    open_mouth = np.array([[0, 1], [1, 0], [3, 0], [4, 1], [3, 2], [1, 2]], dtype=np.float32)
    assert mouth_aspect_ratio(open_mouth) > mouth_aspect_ratio(closed_mouth)


def test_perclos_is_closed_fraction() -> None:
    """PERCLOS equals the fraction of closed frames."""
    assert perclos([True, True, False, False]) == 0.5
    assert perclos([]) == 0.0
    assert perclos([True, True, True]) == 1.0
