"""Video-file demo (Phase 8).

Runs the CabinSense pipeline over a video file and writes an annotated output video
(always-available alternative to the live webcam demo).

Run from the project root:
    python scripts/demo_video.py --input clip.mp4 --output outputs/annotated.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.runtime.build import build_pipeline
from src.runtime.visualize import draw_state


def main() -> None:
    """Annotate a video file with per-frame cabin state."""
    parser = argparse.ArgumentParser(description="Annotate a video with CabinSense")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/annotated.mp4")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        print(f"could not open {args.input}")
        return
    fps = int(capture.get(cv2.CAP_PROP_FPS)) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline = build_pipeline(fps=fps)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            state = pipeline.update(frame)
            writer.write(draw_state(frame, state))
    finally:
        capture.release()
        writer.release()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
