"""Head-pose estimation via PnP (portable core).

Estimates yaw/pitch/roll from five 2D facial landmarks against a canonical 3D face
model using OpenCV solvePnP. OpenCV is available in C++ too, so this stays in the
portable core. Returns degrees in the CabinSense HeadPose convention.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.schemas import HeadPose

# Canonical 3D face model (arbitrary mm units) for the 5 pose landmarks, ordered
# [left_eye_corner, right_eye_corner, nose_tip, left_mouth_corner, right_mouth_corner].
_MODEL_POINTS = np.array(
    [
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [0.0, 0.0, 0.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def estimate_head_pose(image_points: np.ndarray, image_size: tuple[int, int]) -> HeadPose:
    """Estimate head pose (degrees) from 5 landmark points and the image size."""
    if image_points.shape != (5, 2):
        raise ValueError("expected 5 (x, y) pose landmarks")
    width, height = image_size
    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points.astype(np.float64), camera_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return HeadPose(0.0, 0.0, 0.0)
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    pitch, yaw, roll = (float(angle) for angle in cv2.RQDecomp3x3(rotation_matrix)[0])
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)
