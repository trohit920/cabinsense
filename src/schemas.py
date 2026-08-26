"""Shared data structures for the CabinSense pipeline.

Typed records passed between detection, landmark, geometry, classification, and
fusion stages. Kept to standard-library types (no numpy/torch) so it is the
lowest-level module and stays cheap and portable - the fields mirror what a future
C/C++ port would carry as plain structs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates with a detection score."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True)
class ObjectDetection:
    """A localized YOLO detection mapped to a CabinSense attribute label."""

    label: str          # attribute name, e.g. "phone", "drinking"
    box: BBox           # pixel-space box (score field carries the confidence)


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees (yaw = left/right, pitch = up/down, roll = tilt)."""

    yaw: float
    pitch: float
    roll: float


@dataclass
class FrameResult:
    """Raw per-frame predictions before temporal fusion."""

    bbox: BBox | None = None
    ear: float = 0.0                    # eye aspect ratio
    mar: float = 0.0                    # mouth aspect ratio
    head_pose: HeadPose | None = None
    gaze_yaw: float = 0.0               # L2CS gaze yaw (degrees)
    gaze_pitch: float = 0.0            # L2CS gaze pitch (degrees)
    eye_closed_prob: float = 0.0        # learned eye-state classifier
    attribute_probs: dict[str, float] = field(default_factory=dict)  # glasses/cap/...
    object_detections: list[ObjectDetection] = field(default_factory=list)  # YOLO boxes
    gaze_zone: str = "road"
    occupant_count: int = 0
    driver_roi: tuple[int, int, int, int] | None = None  # ROI processed this frame
    full_sweep: bool = False                             # full-frame occupant sweep frame


@dataclass
class CabinState:
    """Final temporally-smoothed cabin state (the deployable output)."""

    driver_state: str = "attentive"     # attentive / drowsy / distracted
    gaze_zone: str = "road"
    yawning: bool = False
    phone: bool = False
    smoking: bool = False
    seatbelt: bool = False
    glasses: bool = False
    cap: bool = False
    drinking: bool = False
    occupant_count: int = 0
    confidences: dict[str, float] = field(default_factory=dict)
