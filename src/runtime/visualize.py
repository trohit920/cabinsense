"""Overlay CabinState onto a video frame for the demos.

Draws the fused driver state, gaze zone, occupant count, and behavior/attribute
flags as a compact text panel. Pure OpenCV drawing.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.schemas import CabinState

_GREEN = (0, 200, 0)
_RED = (0, 0, 255)


def draw_state(frame: np.ndarray, state: CabinState) -> np.ndarray:
    """Draw the CabinState summary onto a copy-free frame (in place) and return it."""
    alert = state.driver_state != "attentive"
    lines = [
        f"driver: {state.driver_state}",
        f"gaze: {state.gaze_zone}   occupants: {state.occupant_count}",
        f"yawn:{int(state.yawning)} phone:{int(state.phone)} smoking:{int(state.smoking)}",
        f"seatbelt:{int(state.seatbelt)} glasses:{int(state.glasses)} cap:{int(state.cap)}",
    ]
    y = 26
    for index, text in enumerate(lines):
        color = _RED if (alert and index == 0) else _GREEN
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 26
    return frame
