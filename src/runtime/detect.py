"""Face / driver-region detection (ONNX YuNet via OpenCV).

Wraps OpenCV's FaceDetectorYN (an ONNX model) to return bounding boxes and an
occupant count. YuNet is small and edge-friendly, and OpenCV's C++ API exposes the
same detector, keeping this deployment-portable.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import DETECT_INPUT_SIZE
from src.schemas import BBox


class FaceDetector:
    """ONNX YuNet face detector returning bounding boxes."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
    ) -> None:
        """Create the YuNet detector with the given thresholds."""
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", DETECT_INPUT_SIZE, score_threshold, nms_threshold, top_k
        )

    def detect(self, image_bgr: np.ndarray) -> list[BBox]:
        """Detect faces in a BGR image; returns a list of bounding boxes."""
        height, width = image_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image_bgr)
        if faces is None:
            return []
        boxes: list[BBox] = []
        for face in faces:
            x, y, w, h = face[:4]
            boxes.append(BBox(float(x), float(y), float(x + w), float(y + h), float(face[-1])))
        return boxes

    def count(self, image_bgr: np.ndarray) -> int:
        """Number of faces detected (occupant-count signal)."""
        return len(self.detect(image_bgr))
