"""Tests for driver-ROI geometry and driver selection (pure logic).

Covers ROI expansion + frame clamping, ROI-local -> full-frame box offsetting,
steering-side driver selection with the largest-face fallback, and drive-side
inference from horizontal position. Deterministic, no I/O.
"""

from __future__ import annotations

from src.config import DriveSide
from src.core.roi import (
    compute_driver_roi,
    infer_drive_side,
    offset_box,
    select_driver,
)
from src.schemas import BBox


def _box(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> BBox:
    """Construct a scored bounding box."""
    return BBox(x1, y1, x2, y2, score)


def test_roi_expands_down_and_clamps() -> None:
    """ROI extends mostly downward (torso/lap) and stays within the frame."""
    face = _box(100, 100, 140, 140)  # 40x40 face
    x1, y1, x2, y2 = compute_driver_roi(face, frame_w=640, frame_h=480, up=0.5, down=4.0, side=1.0)
    assert (x1, y1) == (60, 80)              # 1.0*40 left, 0.5*40 up
    assert x2 == 180                          # 1.0*40 right
    assert y2 == 300                          # 140 + 4.0*40 = 300, under the 480 clamp


def test_roi_clamps_to_frame_bounds() -> None:
    """Expansion never produces coordinates outside the frame."""
    face = _box(10, 10, 50, 50)
    x1, y1, x2, y2 = compute_driver_roi(face, frame_w=200, frame_h=200, up=2.0, down=10.0, side=2.0)
    assert x1 == 0 and y1 == 0
    assert x2 <= 200 and y2 == 200


def test_offset_box_maps_to_full_frame() -> None:
    """A box detected in an ROI crop shifts by the ROI origin."""
    local = _box(5, 5, 15, 25, 0.7)
    shifted = offset_box(local, 100, 200)
    assert (shifted.x1, shifted.y1, shifted.x2, shifted.y2) == (105, 205, 115, 225)
    assert shifted.score == 0.7


def test_select_driver_prefers_steering_side() -> None:
    """For LHD, the left-half face is the driver even if the right one is larger."""
    left = _box(50, 100, 110, 180)        # smaller, left half
    right = _box(400, 100, 520, 240)      # larger, right half
    assert select_driver([left, right], frame_w=640, drive_side=DriveSide.LEFT_HAND) is left
    assert select_driver([left, right], frame_w=640, drive_side=DriveSide.RIGHT_HAND) is right


def test_select_driver_falls_back_to_largest() -> None:
    """With no face on the steering side, the largest face overall is chosen."""
    a = _box(400, 100, 440, 140)
    b = _box(500, 100, 580, 200)          # larger, both on right half
    assert select_driver([a, b], frame_w=640, drive_side=DriveSide.LEFT_HAND) is b


def test_select_driver_empty_is_none() -> None:
    """No faces -> no driver."""
    assert select_driver([], frame_w=640, drive_side=DriveSide.LEFT_HAND) is None


def test_infer_drive_side_from_position() -> None:
    """A driver on the left half implies LHD; right half implies RHD."""
    assert infer_drive_side(120, 640) == DriveSide.LEFT_HAND
    assert infer_drive_side(500, 640) == DriveSide.RIGHT_HAND
