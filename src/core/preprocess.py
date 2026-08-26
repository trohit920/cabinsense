"""Deterministic image preprocessing for model input (pure, C++-portable).

Converts an OpenCV BGR crop into a normalized NCHW float32 tensor exactly as the
training transforms do at eval time. The steps (resize, BGR->RGB, scale, normalize,
HWC->CHW) are simple array ops with direct C++/OpenCV equivalents, and the
constants are recorded in the deployment manifest so a port reproduces them.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import IMAGENET_MEAN, IMAGENET_STD


def preprocess_crop(
    image_bgr: np.ndarray,
    size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Resize/normalize a BGR crop into a (1, 3, size, size) float32 array."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR image")
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)
