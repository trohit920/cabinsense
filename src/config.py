"""Central configuration and named constants for CabinSense.

Every tunable - model ids, input sizes, task/class definitions, geometric and
temporal thresholds, ONNX export settings, and filesystem paths - lives here so
runs are reproducible and free of magic numbers. Import these rather than
hard-coding values elsewhere. All models are open-source and run locally.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Models (local, open-source)
# --------------------------------------------------------------------------- #
BACKBONE: str = "mobilenetv3_small_100"          # timm id (default, edge-friendly)
BACKBONE_TRANSFORMER: str = "mobilevit_xs"       # CNN-vs-Transformer ablation variant
FACE_DETECTOR: str = "yunet"                     # ONNX face detector (OpenCV FaceDetectorYN)
LANDMARK_MODEL: str = "pfld"                     # ONNX 98-point facial landmarks

# --------------------------------------------------------------------------- #
# Input sizes (pixels)
# --------------------------------------------------------------------------- #
DETECT_INPUT_SIZE: tuple[int, int] = (320, 320)  # face detector input (W, H)
FACE_INPUT_SIZE: int = 128                        # face/attribute classifier crop
EYE_INPUT_SIZE: int = 64                          # eye-state classifier crop
UPPER_BODY_INPUT_SIZE: int = 224                  # seatbelt/behavior crop

# Normalization (ImageNet stats; documented in the deployment manifest).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Tasks and classes
# --------------------------------------------------------------------------- #
class EyeState(str, Enum):
    """Binary eye-state classifier labels."""

    OPEN = "open"
    CLOSED = "closed"


class DriverState(str, Enum):
    """Fused, temporally-smoothed high-level driver state."""

    ATTENTIVE = "attentive"
    DROWSY = "drowsy"
    DISTRACTED = "distracted"


class DriveSide(str, Enum):
    """Steering side of the vehicle; sets driver location and gaze-zone mirroring."""

    LEFT_HAND = "lhd"       # driver on the left (e.g. US, EU, Korea)
    RIGHT_HAND = "rhd"      # driver on the right (e.g. India, UK, Japan)


class GazeZone(str, Enum):
    """Coarse gaze regions derived from head pose (+ optional learned gaze)."""

    ROAD = "road"
    LEFT = "left"
    RIGHT = "right"
    REARVIEW = "rearview"
    DASHBOARD = "dashboard"
    CENTER_STACK = "center_stack"
    LAP = "lap"


# Binary attribute/behavior heads defined by the multi-task model (multi-label).
ATTRIBUTE_TASKS: tuple[str, ...] = (
    "glasses", "cap", "seatbelt", "phone", "smoking", "yawn", "drinking",
)
# Heads backed by training data and therefore trusted at inference. Yawn is served by
# MAR geometry (robust, no domain gap) rather than its learned head.
TRAINED_ATTRIBUTES: tuple[str, ...] = (
    "glasses", "cap", "seatbelt", "phone", "smoking", "drinking",
)
# Minimum probability for a trained attribute head to report a positive. Set high to
# suppress false positives - whole-image classifiers over-fire on out-of-domain
# frames. A localized object detector (YOLO) is the better long-term fix for the
# object-like classes (phone, cigarette, bottle/cup, seatbelt, glasses, hat).
ATTRIBUTE_CONFIDENCE_THRESHOLD: float = 0.80

# --------------------------------------------------------------------------- #
# Object detection (COCO-pretrained YOLO) - localized, absence-aware detection for
# object-like classes. Replaces the whole-image classifier heads for classes COCO
# already covers (no training needed). Classes COCO lacks (cigarette, seatbelt, cap)
# still use trained heads until dedicated detectors are trained.
# --------------------------------------------------------------------------- #
OBJECT_DETECTOR_MODEL: str = "yolov8n.pt"       # nano: fast on a 6 GB GPU
OBJECT_DETECTION_CONFIDENCE: float = 0.35        # min box confidence to report
# COCO class name -> CabinSense attribute. A drink is a bottle or a cup.
COCO_TO_ATTRIBUTE: dict[str, str] = {
    "cell phone": "phone",
    "bottle": "drinking",
    "cup": "drinking",
}
# Consolidated custom detector (fine-tuned YOLOv8n) for the object-like classes COCO
# does not cover, trained on merged open detection datasets. One inference serves all
# three (SoC-friendly). scripts/prepare_unified_detection.py + train_object_detector.py.
CUSTOM_DETECTOR_MODEL: str = "custom_yolo.pt"
CUSTOM_DETECTION_CONFIDENCE: float = 0.40
CUSTOM_DETECTOR_CLASSES: tuple[str, ...] = ("smoking", "seatbelt", "cap")
# Case-insensitive source-class aliases -> unified CabinSense class. Lets the ingest
# merge heterogeneous datasets (e.g. Roboflow "Cigarette"/"Vape", the seatbelt set's
# "person-seatbelt") without pre-knowing each export's exact class names. The seatbelt
# negative ("person-noseatbelt") matches nothing and becomes background.
CUSTOM_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "smoking": ("cigarette", "cigarettes", "vape", "smoke", "smoking", "cig"),
    "seatbelt": ("seatbelt", "seat-belt", "seat_belt", "person-seatbelt", "belt"),
    "cap": ("cap", "caps", "hat", "baseball cap", "headwear", "beanie"),
}
# Detector class name -> attribute name (identity: the unified classes are the attrs).
CUSTOM_TO_ATTRIBUTE: dict[str, str] = {c: c for c in CUSTOM_DETECTOR_CLASSES}

# Attributes served by a localized object detector; these override any classifier head.
DETECTOR_ATTRIBUTES: tuple[str, ...] = ("phone", "drinking", "smoking", "seatbelt", "cap")
# Face-attribute served by the whole-image classifier. Glasses is a genuine face
# attribute (not a separable object), so the face-crop classifier is trustworthy.
CLASSIFIER_ATTRIBUTES: tuple[str, ...] = ("glasses",)
# Object-like classes with no localized detector yet. Empty now that smoking/seatbelt/
# cap have a trained detector; kept for the fusion fallback contract.
PENDING_DETECTOR_ATTRIBUTES: tuple[str, ...] = ()

# --------------------------------------------------------------------------- #
# Geometric thresholds (paper-based baselines; tuned during evaluation)
# --------------------------------------------------------------------------- #
EAR_CLOSED_THRESHOLD: float = 0.20        # eye aspect ratio below this = closed
MAR_YAWN_THRESHOLD: float = 0.60          # mouth aspect ratio above this = yawn
DISTRACTION_YAW_DEGREES: float = 35.0     # |yaw| beyond this = looking away
DISTRACTION_PITCH_DEGREES: float = 25.0   # |pitch| beyond this = looking away

# Gaze-zone yaw/pitch bin edges (degrees) for the geometry baseline.
GAZE_YAW_BINS_DEGREES: tuple[float, ...] = (-45.0, -15.0, 15.0, 45.0)
GAZE_PITCH_DOWN_DEGREES: float = -20.0    # below this pitch = looking down (lap/dashboard)

# --------------------------------------------------------------------------- #
# Driver ROI + two-rate scheduling (SoC efficiency)
# --------------------------------------------------------------------------- #
# Most frames only process a region of interest around the driver (face -> torso ->
# lap, where phone/drink/cigarette/seatbelt/cap appear); a full-frame sweep runs every
# ROI_FULL_SWEEP_SECONDS to recount occupants and re-localize the driver.
DEFAULT_DRIVE_SIDE: DriveSide = DriveSide.LEFT_HAND
ROI_FULL_SWEEP_SECONDS: float = 3.0        # cadence of the full-frame occupant sweep
ROI_EXPAND_UP: float = 0.6                 # face heights added above (hair/cap)
ROI_EXPAND_DOWN: float = 4.0               # face heights added below (torso/lap)
ROI_EXPAND_SIDE: float = 1.5               # face widths added each side (hands/objects)
DRIVE_SIDE_AUTOINIT_SAMPLES: int = 5       # full-sweep votes before locking drive side

# --------------------------------------------------------------------------- #
# Temporal fusion (seconds; frame counts derived from the stream FPS at runtime)
# --------------------------------------------------------------------------- #
PERCLOS_WINDOW_SECONDS: float = 30.0      # window for percent-eye-closure
PERCLOS_DROWSY_RATIO: float = 0.15        # closed-fraction above this = drowsy
# Probability above which the learned eye-state classifier calls a frame "closed".
# Its output is then accumulated in its own PERCLOS-style window rather than being
# trusted per-frame: a single frame over threshold is a blink, not drowsiness.
DROWSY_MODEL_CLOSED_PROB: float = 0.5
DISTRACTION_WINDOW_SECONDS: float = 2.0   # sustained off-road gaze to flag distraction
DISTRACTION_OFF_ROAD_RATIO: float = 0.6   # off-road fraction in the window to flag distraction
SMOOTHING_WINDOW_SECONDS: float = 1.0     # majority-vote window for per-frame states
HYSTERESIS_ON_RATIO: float = 0.6          # fraction to switch a flag ON
HYSTERESIS_OFF_RATIO: float = 0.4         # fraction to switch a flag OFF
MIN_CONFIDENCE: float = 0.55              # below this a per-frame prediction abstains

# --------------------------------------------------------------------------- #
# ONNX export / quantization
# --------------------------------------------------------------------------- #
ONNX_OPSET: int = 17
ONNX_DYNAMIC_BATCH: bool = False          # static shapes for SoC portability

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
MLRUNS_DIR: Path = OUTPUT_DIR / "mlruns"
DEPLOY_DIR: Path = PROJECT_ROOT / "deploy"
DEPLOY_MODELS_DIR: Path = DEPLOY_DIR / "models"
