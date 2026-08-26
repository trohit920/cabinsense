"""COCO-pretrained YOLO object detector for object-like cabin attributes.

Runs a small COCO YOLO model on the full frame and maps the classes CabinSense
cares about (cell phone -> phone, bottle/cup -> drinking) into labeled, localized
ObjectDetection boxes. Localized detection with a real absence concept replaces the
whole-image attribute classifiers for these classes, which over-fired on
out-of-domain cabin frames. Torch runs on GPU when available; ONNX export is a
future step for SoC portability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import (
    COCO_TO_ATTRIBUTE,
    DEPLOY_MODELS_DIR,
    OBJECT_DETECTION_CONFIDENCE,
    OBJECT_DETECTOR_MODEL,
)
from src.schemas import BBox, ObjectDetection


class ObjectDetector:
    """Wraps a COCO YOLO model and emits CabinSense-labeled detections."""

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = OBJECT_DETECTION_CONFIDENCE,
        class_map: dict[str, str] | None = None,
    ) -> None:
        """Load the YOLO model, resolving/downloading weights under the models dir."""
        import torch
        from ultralytics import YOLO  # lazy: heavy import, optional dependency

        path = model_path or (DEPLOY_MODELS_DIR / OBJECT_DETECTOR_MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads a recognized weight name to the given path if absent.
        self._model = YOLO(str(path) if path.exists() else OBJECT_DETECTOR_MODEL)
        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._confidence = confidence
        self._class_map = class_map or COCO_TO_ATTRIBUTE
        self._names: dict[int, str] = self._model.names
        # COCO class ids we care about, so YOLO can skip the rest.
        self._keep_ids = [i for i, n in self._names.items() if n in self._class_map]

    def detect(self, frame_bgr: np.ndarray) -> list[ObjectDetection]:
        """Return labeled detections for the mapped classes in one frame."""
        if not self._keep_ids:
            return []
        results = self._model.predict(
            frame_bgr,
            conf=self._confidence,
            classes=self._keep_ids,
            device=self._device,
            verbose=False,
        )
        detections: list[ObjectDetection] = []
        for result in results:
            for box in result.boxes:
                coco_name = self._names[int(box.cls[0])]
                label = self._class_map.get(coco_name)
                if label is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                score = float(box.conf[0])
                detections.append(
                    ObjectDetection(label=label, box=BBox(x1, y1, x2, y2, score))
                )
        return detections
