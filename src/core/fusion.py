"""Multi-signal fusion into a single driver state (pure, C++-portable).

Combines geometry (EAR, PERCLOS, head pose/gaze) with learned signals (eye-state
probability) into per-frame decisions, then a single high-level driver state. This
is the "final decision quality when combining multiple models" logic, kept as
pure functions.
"""

from __future__ import annotations

from src.config import (
    EAR_CLOSED_THRESHOLD,
    PERCLOS_DROWSY_RATIO,
    DriverState,
    GazeZone,
)


def eye_closed(ear: float, eye_closed_prob: float, ear_threshold: float = EAR_CLOSED_THRESHOLD) -> bool:
    """Fuse geometry and the learned eye-state model into a closed/open decision."""
    return ear < ear_threshold or eye_closed_prob >= 0.5


def is_drowsy(perclos_value: float, perclos_threshold: float = PERCLOS_DROWSY_RATIO) -> bool:
    """Drowsy when the closed-eye fraction over the window exceeds the threshold."""
    return perclos_value >= perclos_threshold


def is_distracted(off_road_ratio: float, distraction_threshold: float) -> bool:
    """Distracted when the gaze has been off the road for a sustained fraction."""
    return off_road_ratio >= distraction_threshold


def gaze_off_road(gaze_zone: str) -> bool:
    """True when the current gaze zone is anything other than the road ahead."""
    return gaze_zone != GazeZone.ROAD.value


def fuse_driver_state(drowsy: bool, distracted: bool) -> str:
    """Collapse the boolean signals into one driver state (drowsy takes priority)."""
    if drowsy:
        return DriverState.DROWSY.value
    if distracted:
        return DriverState.DISTRACTED.value
    return DriverState.ATTENTIVE.value
