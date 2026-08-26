"""Tests for attribute-flag precedence in the CabinSense pipeline.

Verifies that object-like classes are driven by the localized YOLO detector (with a
real absence concept) and take precedence over the whole-image classifier heads,
while non-detector trained heads still use the classifier threshold. Uses fps=1 so
the smoothing/hysteresis window is a single frame and flags flip immediately.
"""

from __future__ import annotations

import pytest

from src.config import ATTRIBUTE_CONFIDENCE_THRESHOLD
from src.runtime.pipeline import CabinSensePipeline
from src.schemas import BBox, FrameResult, ObjectDetection


@pytest.fixture
def pipeline() -> CabinSensePipeline:
    """Pipeline with placeholder detector/landmarker (unused by _update_flags)."""
    return CabinSensePipeline(detector=object(), landmarker=object(), fps=1)


def _phone_detection() -> ObjectDetection:
    """A single high-confidence phone detection box."""
    return ObjectDetection(label="phone", box=BBox(10.0, 10.0, 40.0, 60.0, 0.9))


def test_detector_presence_sets_flag(pipeline: CabinSensePipeline) -> None:
    """A YOLO phone detection turns the phone flag on."""
    frame = FrameResult(object_detections=[_phone_detection()])
    assert pipeline._update_flags(frame)["phone"] is True


def test_seatbelt_detector_presence_sets_flag(pipeline: CabinSensePipeline) -> None:
    """A seatbelt detection (custom detector) turns the seatbelt flag on."""
    det = ObjectDetection(label="seatbelt", box=BBox(5.0, 5.0, 50.0, 90.0, 0.8))
    frame = FrameResult(object_detections=[det])
    assert pipeline._update_flags(frame)["seatbelt"] is True


def test_detector_class_ignores_classifier_prob(pipeline: CabinSensePipeline) -> None:
    """Cap is detector-backed, so a confident classifier prob alone does not fire it."""
    frame = FrameResult(attribute_probs={"cap": 0.99})
    assert pipeline._update_flags(frame)["cap"] is False


def test_detector_absence_overrides_classifier(pipeline: CabinSensePipeline) -> None:
    """No phone detection keeps phone off even when the classifier head is confident."""
    frame = FrameResult(attribute_probs={"phone": 0.99}, object_detections=[])
    assert pipeline._update_flags(frame)["phone"] is False


def test_non_detector_head_uses_classifier(pipeline: CabinSensePipeline) -> None:
    """Glasses (no detector) fires from the classifier head above the threshold."""
    prob = ATTRIBUTE_CONFIDENCE_THRESHOLD + 0.05
    frame = FrameResult(attribute_probs={"glasses": prob})
    assert pipeline._update_flags(frame)["glasses"] is True


def test_non_detector_head_below_threshold_stays_off(pipeline: CabinSensePipeline) -> None:
    """Glasses stays off when the classifier probability is below the threshold."""
    prob = ATTRIBUTE_CONFIDENCE_THRESHOLD - 0.05
    frame = FrameResult(attribute_probs={"glasses": prob})
    assert pipeline._update_flags(frame)["glasses"] is False
