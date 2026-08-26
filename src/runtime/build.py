"""Pipeline assembly from on-disk model artifacts.

Builds a CabinSensePipeline from the ONNX models in the deploy directory: the YuNet
detector and PFLD landmarks are required; the trained drowsiness and multi-task
models are optional (the geometry-only pipeline runs without them). Shared by the
CLI demos and the evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path

from src.config import DEFAULT_DRIVE_SIDE, DEPLOY_MODELS_DIR, DriveSide
from src.runtime.detect import FaceDetector
from src.runtime.landmarks import LandmarkDetector
from src.runtime.onnx_session import OnnxModel
from src.runtime.pipeline import CabinSensePipeline


def _optional(path: Path) -> OnnxModel | None:
    """Load an OnnxModel if the file exists, else None."""
    return OnnxModel(path) if path.exists() else None


def build_pipeline(
    models_dir: Path = DEPLOY_MODELS_DIR,
    fps: int = 30,
    use_headpose_gaze: bool = True,
    use_object_detector: bool = True,
    drive_side: DriveSide = DEFAULT_DRIVE_SIDE,
    autoinit_drive_side: bool = True,
) -> CabinSensePipeline:
    """Assemble the pipeline from artifacts in ``models_dir``.

    When ``use_headpose_gaze`` is set, adds 6DRepNet head-pose and L2CS gaze
    estimators (PyTorch); disable it for the lightweight ONNX-only geometry path.
    When ``use_object_detector`` is set, adds a COCO YOLO detector that serves the
    object-like classes (phone, drinking) with real localized boxes.
    """
    detector_path = models_dir / "yunet.onnx"
    landmark_path = models_dir / "pfld.onnx"
    for required in (detector_path, landmark_path):
        if not required.exists():
            raise FileNotFoundError(f"required model missing: {required} (run scripts/download_models.py)")
    detector = FaceDetector(detector_path)
    landmarker = LandmarkDetector(OnnxModel(landmark_path))
    drowsiness = _optional(models_dir / "drowsiness.onnx")
    multitask = _optional(models_dir / "multitask.onnx")

    headpose = gaze = None
    if use_headpose_gaze:
        from src.runtime.gaze_l2cs import GazeEstimator
        from src.runtime.headpose import HeadPoseEstimator

        headpose = HeadPoseEstimator()
        gaze = GazeEstimator()

    object_detectors: list = []
    if use_object_detector:
        from src.config import (
            CUSTOM_DETECTION_CONFIDENCE,
            CUSTOM_DETECTOR_MODEL,
            CUSTOM_TO_ATTRIBUTE,
            OBJECT_DETECTOR_MODEL,
        )
        from src.runtime.object_detect import ObjectDetector

        # COCO-pretrained detector for phone/drinking (classes COCO already covers).
        object_detectors.append(ObjectDetector(models_dir / OBJECT_DETECTOR_MODEL))
        # Consolidated custom detector for smoking/seatbelt/cap (one inference).
        custom_path = models_dir / CUSTOM_DETECTOR_MODEL
        if custom_path.exists():
            object_detectors.append(
                ObjectDetector(
                    custom_path,
                    confidence=CUSTOM_DETECTION_CONFIDENCE,
                    class_map=CUSTOM_TO_ATTRIBUTE,
                )
            )
    return CabinSensePipeline(
        detector, landmarker, drowsiness, multitask, headpose, gaze, object_detectors,
        fps=fps, drive_side=drive_side, autoinit_drive_side=autoinit_drive_side,
    )
