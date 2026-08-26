"""Unit tests for gaze-zone mapping and multi-signal fusion."""

from __future__ import annotations

from src.config import DriverState, GazeZone
from src.core.fusion import eye_closed, fuse_driver_state, gaze_off_road, is_drowsy
from src.core.gaze import head_pose_to_zone


def test_gaze_zone_center_is_road() -> None:
    """Looking straight ahead maps to the road zone."""
    assert head_pose_to_zone(yaw=0.0, pitch=0.0) == GazeZone.ROAD.value


def test_gaze_zone_left_right_and_down() -> None:
    """Large yaw maps to sides; strong downward pitch maps to the dashboard."""
    assert head_pose_to_zone(yaw=-60.0, pitch=0.0) == GazeZone.LEFT.value
    assert head_pose_to_zone(yaw=60.0, pitch=0.0) == GazeZone.RIGHT.value
    assert head_pose_to_zone(yaw=0.0, pitch=-40.0) == GazeZone.DASHBOARD.value


def test_gaze_zone_rhd_mirror_swaps_sides() -> None:
    """RHD mirroring swaps the side-dependent zones (rearview <-> center-stack)."""
    assert head_pose_to_zone(yaw=-30.0, pitch=0.0) == GazeZone.REARVIEW.value
    assert head_pose_to_zone(yaw=-30.0, pitch=0.0, mirror=True) == GazeZone.CENTER_STACK.value
    assert head_pose_to_zone(yaw=30.0, pitch=0.0, mirror=True) == GazeZone.REARVIEW.value


def test_eye_closed_fuses_geometry_and_model() -> None:
    """An eye is closed if EAR is low OR the learned model says closed."""
    assert eye_closed(ear=0.1, eye_closed_prob=0.0) is True   # geometry
    assert eye_closed(ear=0.4, eye_closed_prob=0.9) is True   # learned
    assert eye_closed(ear=0.4, eye_closed_prob=0.1) is False


def test_driver_state_priority() -> None:
    """Drowsy takes priority over distracted, which beats attentive."""
    assert fuse_driver_state(drowsy=True, distracted=True) == DriverState.DROWSY.value
    assert fuse_driver_state(drowsy=False, distracted=True) == DriverState.DISTRACTED.value
    assert fuse_driver_state(drowsy=False, distracted=False) == DriverState.ATTENTIVE.value


def test_is_drowsy_and_gaze_off_road() -> None:
    """PERCLOS threshold and off-road detection behave as expected."""
    assert is_drowsy(0.2, perclos_threshold=0.15) is True
    assert is_drowsy(0.1, perclos_threshold=0.15) is False
    assert gaze_off_road(GazeZone.LEFT.value) is True
    assert gaze_off_road(GazeZone.ROAD.value) is False
