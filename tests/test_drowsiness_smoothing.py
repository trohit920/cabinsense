"""Tests that both eye-closure signals are accumulated before raising DROWSY.

The geometry branch (EAR/PERCLOS) and the learned-classifier branch must be smoothed
over the same window: a single frame above the classifier threshold is a blink, not
drowsiness, and must not latch the flag.
"""

from __future__ import annotations

import pytest

from src.config import (
    DROWSY_MODEL_CLOSED_PROB,
    PERCLOS_DROWSY_RATIO,
    PERCLOS_WINDOW_SECONDS,
    DriverState,
)
from src.runtime.pipeline import CabinSensePipeline, _window
from src.schemas import BBox, FrameResult

FPS = 30
WINDOW_FRAMES = _window(PERCLOS_WINDOW_SECONDS, FPS)
OPEN_EAR = 0.30          # comfortably above EAR_CLOSED_THRESHOLD
CONFIDENT_PROB = DROWSY_MODEL_CLOSED_PROB + 0.03


@pytest.fixture
def pipeline() -> CabinSensePipeline:
    """Pipeline with placeholder models; only the fusion path is exercised."""
    return CabinSensePipeline(detector=object(), landmarker=object(), fps=FPS)


def _frame(eye_closed_prob: float, ear: float = OPEN_EAR) -> FrameResult:
    """A forward-facing single-occupant frame with the given classifier output."""
    return FrameResult(
        bbox=BBox(10.0, 10.0, 50.0, 70.0, 0.9),
        ear=ear,
        gaze_zone="road",
        eye_closed_prob=eye_closed_prob,
        occupant_count=1,
    )


def test_single_confident_frame_does_not_latch_drowsy(pipeline: CabinSensePipeline) -> None:
    """One frame over the classifier threshold is a blink and must not fire DROWSY."""
    state = pipeline._fuse(_frame(CONFIDENT_PROB))
    assert state.driver_state != DriverState.DROWSY.value


def test_sustained_classifier_closure_fires_drowsy(pipeline: CabinSensePipeline) -> None:
    """Sustained classifier closure past the PERCLOS ratio does fire DROWSY."""
    needed = int(PERCLOS_DROWSY_RATIO * WINDOW_FRAMES) + 2
    for _ in range(needed):
        state = pipeline._fuse(_frame(CONFIDENT_PROB))
    assert state.driver_state == DriverState.DROWSY.value


def test_open_eyes_stay_alert(pipeline: CabinSensePipeline) -> None:
    """A long run of open-eyed, forward-facing frames never fires DROWSY."""
    for _ in range(WINDOW_FRAMES):
        state = pipeline._fuse(_frame(0.05))
    assert state.driver_state != DriverState.DROWSY.value
