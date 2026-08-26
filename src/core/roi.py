"""Driver region-of-interest geometry and driver selection (pure, C++-portable).

Derives the driver ROI (face -> torso -> lap) from a detected face box, selects which
detected face is the driver given the vehicle's steering side, offsets ROI-local boxes
back to full-frame coordinates, and infers steering side from the driver's position.
Pure arithmetic on plain structs so it ports directly to an on-device runtime.
"""

from __future__ import annotations

from src.config import (
    ROI_EXPAND_DOWN,
    ROI_EXPAND_SIDE,
    ROI_EXPAND_UP,
    DriveSide,
)
from src.schemas import BBox

Roi = tuple[int, int, int, int]


def _area(box: BBox) -> float:
    """Pixel area of a bounding box."""
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def compute_driver_roi(
    face: BBox,
    frame_w: int,
    frame_h: int,
    up: float = ROI_EXPAND_UP,
    down: float = ROI_EXPAND_DOWN,
    side: float = ROI_EXPAND_SIDE,
) -> Roi:
    """Expand a driver face box into a face->torso->lap ROI, clamped to the frame."""
    face_w, face_h = face.x2 - face.x1, face.y2 - face.y1
    x1 = max(0, int(face.x1 - side * face_w))
    y1 = max(0, int(face.y1 - up * face_h))
    x2 = min(frame_w, int(face.x2 + side * face_w))
    y2 = min(frame_h, int(face.y2 + down * face_h))
    return x1, y1, x2, y2


def offset_box(box: BBox, off_x: int, off_y: int) -> BBox:
    """Shift a box detected in an ROI crop back into full-frame coordinates."""
    return BBox(box.x1 + off_x, box.y1 + off_y, box.x2 + off_x, box.y2 + off_y, box.score)


def select_driver(boxes: list[BBox], frame_w: int, drive_side: DriveSide) -> BBox | None:
    """Pick the driver: the largest face on the steering-side half of the frame.

    Falls back to the largest face overall when no face sits on the expected side
    (e.g. a turned/leaning driver), so detection is never lost to the side heuristic.
    """
    if not boxes:
        return None
    center = frame_w / 2.0
    if drive_side == DriveSide.LEFT_HAND:
        side_boxes = [b for b in boxes if (b.x1 + b.x2) / 2.0 <= center]
    else:
        side_boxes = [b for b in boxes if (b.x1 + b.x2) / 2.0 >= center]
    return max(side_boxes or boxes, key=_area)


def infer_drive_side(face_center_x: float, frame_w: int) -> DriveSide:
    """Guess steering side from the driver's horizontal position in the frame."""
    return DriveSide.LEFT_HAND if face_center_x <= frame_w / 2.0 else DriveSide.RIGHT_HAND
