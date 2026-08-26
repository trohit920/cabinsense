"""Geometric drowsiness/yawn features (pure, C++-portable).

Eye Aspect Ratio (EAR, Soukupova & Cech 2016), Mouth Aspect Ratio (MAR), and
PERCLOS - computed from explicit landmark point arrays so the math is independent
of any specific landmark model's indexing (the index mapping lives in
``landmark_indices``). Pure numpy with direct C++ equivalents.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(a - b))


def _aspect_ratio(points: np.ndarray) -> float:
    """Six-point aspect ratio: (|p1-p5| + |p2-p4|) / (2 |p0-p3|).

    Points are ordered [left_corner, top1, top2, right_corner, bottom2, bottom1],
    matching the classic EAR layout; the same formula serves the mouth (MAR).
    """
    if points.shape != (6, 2):
        raise ValueError("expected 6 (x, y) points")
    horizontal = _euclidean(points[0], points[3])
    if horizontal < 1e-6:
        return 0.0
    vertical = _euclidean(points[1], points[5]) + _euclidean(points[2], points[4])
    return vertical / (2.0 * horizontal)


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """EAR for one eye from its 6 contour points; low values indicate closure."""
    return _aspect_ratio(eye_points)


def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """MAR from 6 mouth points; high values indicate an open mouth / yawn."""
    return _aspect_ratio(mouth_points)


def perclos(closed_flags: Sequence[bool]) -> float:
    """Fraction of frames with eyes closed over a window (PERCLOS)."""
    if not closed_flags:
        return 0.0
    return float(np.mean([1.0 if flag else 0.0 for flag in closed_flags]))
