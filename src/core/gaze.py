"""Gaze-zone estimation from head pose (geometry baseline, pure).

Maps head yaw/pitch to a coarse cabin gaze zone. This is the always-available
baseline (no labeled gaze dataset required); a learned gaze model can later refine
it behind the same interface. Pure arithmetic, C++-portable.
"""

from __future__ import annotations

from src.config import (
    GAZE_PITCH_DOWN_DEGREES,
    GAZE_YAW_BINS_DEGREES,
    GazeZone,
)


def head_pose_to_zone(
    yaw: float,
    pitch: float,
    yaw_bins: tuple[float, ...] = GAZE_YAW_BINS_DEGREES,
    pitch_down: float = GAZE_PITCH_DOWN_DEGREES,
    mirror: bool = False,
) -> str:
    """Map yaw/pitch (degrees) to a gaze zone value.

    Looking clearly down maps to the dashboard/console; otherwise the yaw bins
    (default -45/-15/15/45) split into left, rearview, road (center), center-stack,
    and right. The bins assume a left-hand-drive cabin (mirror/center-stack on the
    driver's right); ``mirror=True`` negates yaw for a right-hand-drive cabin, which
    is the mirror image, swapping the side-dependent zones.
    """
    if pitch <= pitch_down:
        return GazeZone.DASHBOARD.value
    if mirror:
        yaw = -yaw
    left_far, left_near, right_near, right_far = yaw_bins
    if yaw < left_far:
        return GazeZone.LEFT.value
    if yaw < left_near:
        return GazeZone.REARVIEW.value
    if yaw <= right_near:
        return GazeZone.ROAD.value
    if yaw <= right_far:
        return GazeZone.CENTER_STACK.value
    return GazeZone.RIGHT.value
