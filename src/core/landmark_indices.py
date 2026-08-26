"""Landmark index maps for the 98-point (WFLW/PFLD) model (pure).

Selects and reorders raw landmark points into the 6-point arrays the EAR/MAR
features expect, in the order [left_corner, top1, top2, right_corner, bottom2,
bottom1]. Indices assume the WFLW-98 convention used by common PFLD exports and are
re-validated when the real ONNX landmark model is wired in (Phase 4).
"""

from __future__ import annotations

import numpy as np

# WFLW-98 landmark indices (validated against the deployed model in Phase 4).
LEFT_EYE_IDX: tuple[int, ...] = (60, 61, 63, 64, 65, 67)
RIGHT_EYE_IDX: tuple[int, ...] = (68, 69, 71, 72, 73, 75)
MOUTH_IDX: tuple[int, ...] = (76, 78, 80, 82, 84, 86)
# 5-point subset (eye corners, nose tip, mouth corners) for solvePnP head pose.
POSE_IDX: tuple[int, ...] = (60, 72, 54, 76, 82)


def _select(landmarks: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Return the (len(indices), 2) subset of landmark points."""
    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError("landmarks must be an (N, 2) array")
    return landmarks[list(indices)]


def left_eye_points(landmarks: np.ndarray) -> np.ndarray:
    """Six left-eye points in EAR order."""
    return _select(landmarks, LEFT_EYE_IDX)


def right_eye_points(landmarks: np.ndarray) -> np.ndarray:
    """Six right-eye points in EAR order."""
    return _select(landmarks, RIGHT_EYE_IDX)


def mouth_points(landmarks: np.ndarray) -> np.ndarray:
    """Six mouth points in MAR order."""
    return _select(landmarks, MOUTH_IDX)


def pose_points(landmarks: np.ndarray) -> np.ndarray:
    """Five points used for solvePnP head-pose estimation."""
    return _select(landmarks, POSE_IDX)
