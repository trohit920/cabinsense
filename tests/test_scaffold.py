"""Scaffold tests: validate the pure foundation (config + schemas).

Avoids importing model-dependent modules so it runs before/without the heavy
deep-learning stack. Phase-specific modules get their own tests as implemented.
"""

from __future__ import annotations

from src import config
from src.schemas import BBox, CabinState, FrameResult, HeadPose


def test_seed_is_fixed() -> None:
    """Reproducibility seed is the project-wide constant 42."""
    assert config.SEED == 42


def test_attribute_tasks_present() -> None:
    """The multi-task attribute set covers the planned binary heads."""
    for task in ("glasses", "cap", "seatbelt", "phone", "smoking", "yawn"):
        assert task in config.ATTRIBUTE_TASKS


def test_gaze_zones_are_unique() -> None:
    """Gaze-zone enum values are distinct."""
    values = [zone.value for zone in config.GazeZone]
    assert len(values) == len(set(values))


def test_thresholds_in_sane_ranges() -> None:
    """Geometric and temporal thresholds sit in plausible ranges."""
    assert 0.0 < config.EAR_CLOSED_THRESHOLD < 1.0
    assert 0.0 < config.MAR_YAWN_THRESHOLD < 2.0
    assert 0.0 < config.PERCLOS_DROWSY_RATIO < 1.0
    assert config.HYSTERESIS_OFF_RATIO < config.HYSTERESIS_ON_RATIO


def test_schema_construction() -> None:
    """Core records build with sensible defaults and hold their fields."""
    frame = FrameResult(bbox=BBox(0, 0, 10, 10, 0.9), head_pose=HeadPose(0.0, 0.0, 0.0))
    assert frame.occupant_count == 0
    state = CabinState(driver_state="drowsy", occupant_count=1)
    assert state.driver_state == "drowsy"
    assert state.yawning is False
