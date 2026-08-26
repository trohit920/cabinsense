"""Live webcam demo (Phase 8).

Runs the CabinSense pipeline on the default camera and overlays the cabin state.
Press 'q' to quit. Requires the detector/landmark models (and optionally the trained
drowsiness/multi-task models) in deploy/models/.

Run from the project root:
    python scripts/demo_webcam.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.runtime.build import build_pipeline
from src.runtime.visualize import draw_state


def main() -> None:
    """Open the webcam and run the pipeline until 'q' is pressed."""
    pipeline = build_pipeline(fps=30)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("could not open webcam")
        return
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            state = pipeline.update(frame)
            cv2.imshow("CabinSense", draw_state(frame, state))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
