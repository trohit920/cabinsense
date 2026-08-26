"""Facial landmark detection (ONNX PFLD, 98-point).

Runs a PFLD ONNX model on a face crop and returns 98 landmark points in the
original image's pixel coordinates. Preprocessing (112x112 RGB, [0,1] scale) matches
the common PFLD export and is recorded in the deployment manifest; it is re-validated
when the concrete model is wired in.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.runtime.onnx_session import OnnxModel
from src.schemas import BBox

PFLD_INPUT_SIZE: int = 112
NUM_LANDMARKS: int = 98


class LandmarkDetector:
    """PFLD ONNX landmark predictor."""

    def __init__(self, model: OnnxModel, input_size: int = PFLD_INPUT_SIZE) -> None:
        """Bind a loaded PFLD OnnxModel and its expected square input size."""
        self.model = model
        self.input_size = input_size

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Resize/scale a face crop into the PFLD (1, 3, S, S) float32 input."""
        resized = cv2.resize(crop_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0).astype(np.float32)

    def predict(self, image_bgr: np.ndarray, box: BBox) -> np.ndarray:
        """Return 98 landmarks in image pixel coordinates for the face in ``box``."""
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = int(box.x2), int(box.y2)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
        raw = self.model.run(self._preprocess(crop))[0].reshape(-1, 2)  # normalized [0, 1]
        scaled = raw[:NUM_LANDMARKS].copy()
        scaled[:, 0] = x1 + scaled[:, 0] * (x2 - x1)
        scaled[:, 1] = y1 + scaled[:, 1] * (y2 - y1)
        return scaled.astype(np.float32)
