"""Run the full upgraded pipeline on a video and save an annotated DMS demo.

Draws the face box, 6DRepNet head-pose axis, L2CS gaze arrow, and a temporal state
panel (driver state, gaze zone, head/gaze angles, EAR/PERCLOS, attributes, occupant
count) per frame. Works on any clip; defaults to the downloaded in-cabin sample.

Run from the project root:
    python scripts/demo_dms_video.py --input data/videos/incabin_sample.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.config import EAR_CLOSED_THRESHOLD, OUTPUT_DIR, DriveSide
from src.core.landmark_indices import left_eye_points, mouth_points, right_eye_points
from src.runtime.build import build_pipeline
from src.schemas import CabinState, FrameResult, HeadPose

OUT_SIZE = (1280, 720)
GREEN, RED, YELLOW, CYAN, WHITE = (0, 200, 0), (0, 0, 235), (0, 210, 210), (235, 235, 0), (255, 255, 255)
_ALERT_COLORS = {"attentive": GREEN, "drowsy": RED, "distracted": YELLOW}


def _draw_gaze(frame, cx: int, cy: int, yaw_deg: float, pitch_deg: float, length: int = 160) -> None:
    """Draw the L2CS gaze direction as an arrow from the face center."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    dx = int(-length * math.sin(yaw) * math.cos(pitch))
    dy = int(-length * math.sin(pitch))
    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), CYAN, 3, tipLength=0.25)


def _panel(frame, result: FrameResult, state: CabinState) -> None:
    """Draw the DMS readout panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, 372), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    hp = result.head_pose or HeadPose(0.0, 0.0, 0.0)  # no-driver frame: show zeros
    lines = [
        (f"STATUS: {state.driver_state.upper()}", _ALERT_COLORS.get(state.driver_state, WHITE)),
        (f"gaze zone : {state.gaze_zone}", WHITE),
        (f"head pose : yaw {hp.yaw:+.0f}  pitch {hp.pitch:+.0f}", WHITE),
        (f"gaze      : yaw {result.gaze_yaw:+.0f}  pitch {result.gaze_pitch:+.0f}", WHITE),
        (f"EAR {result.ear:.2f}  PERCLOS {state.confidences.get('perclos', 0):.2f}", WHITE),
        (f"drowsy p {result.eye_closed_prob:.2f}  off-road {state.confidences.get('off_road', 0):.2f}", WHITE),
        (f"glasses {int(state.glasses)}  cap {int(state.cap)}  seatbelt {int(state.seatbelt)}", WHITE),
        (f"phone {int(state.phone)}  smoking {int(state.smoking)}  drink {int(state.drinking)}  yawn {int(state.yawning)}", WHITE),
        (f"occupants : {state.occupant_count}", WHITE),
    ]
    y = 34
    for text, color in lines:
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 36


_CLASS_COLORS = {
    "glasses": (0, 200, 0), "cap": (0, 165, 255), "seatbelt": (255, 150, 0),
    "phone": (0, 0, 235), "smoking": (0, 0, 150), "drinking": (200, 0, 200), "yawn": (0, 210, 210),
}
# Approximate region (fractions of the face box) for the classifier/geometry classes
# that lack a localized detector. Detector-backed classes (phone, drinking) draw their
# real YOLO boxes instead.
_CLASS_REGIONS = {
    "glasses": (0.0, 0.22, 1.0, 0.48),
    "yawn": (0.25, 0.60, 0.75, 0.95),
}


def _draw_detections(frame, result) -> None:
    """Draw the real YOLO boxes (phone, drinking) with class label and confidence."""
    for det in result.object_detections:
        color = _CLASS_COLORS.get(det.label, WHITE)
        b = det.box
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        label = f"{det.label.upper()} {b.score:.2f}"
        cv2.putText(frame, label, (int(b.x1), max(12, int(b.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_class_boxes(frame, box, state) -> None:
    """Draw approximate region boxes for the non-detector attributes that are flagged."""
    height, width = frame.shape[:2]
    box_w, box_h = box.x2 - box.x1, box.y2 - box.y1
    active = {"glasses": state.glasses, "yawn": state.yawning}
    for name, is_on in active.items():
        if not is_on:
            continue
        fx1, fy1, fx2, fy2 = _CLASS_REGIONS[name]
        x1 = int(max(0, box.x1 + fx1 * box_w))
        y1 = int(max(0, box.y1 + fy1 * box_h))
        x2 = int(min(width, box.x1 + fx2 * box_w))
        y2 = int(min(height, box.y1 + fy2 * box_h))
        color = _CLASS_COLORS[name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name.upper(), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_landmarks(pipeline, frame, box) -> None:
    """Overlay the 98 landmarks, highlighting eyes (green) and mouth (red)."""
    pts = pipeline.landmarker.predict(frame, box)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 1, CYAN, -1)
    for group, color in ((left_eye_points(pts), GREEN), (right_eye_points(pts), GREEN), (mouth_points(pts), RED)):
        for x, y in group:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def _draw_roi(pipeline, frame, result: FrameResult) -> None:
    """Draw the driver ROI box and the current mode (full sweep vs ROI) + drive side."""
    if result.driver_roi is None:
        return
    rx1, ry1, rx2, ry2 = result.driver_roi
    color = YELLOW if result.full_sweep else (150, 150, 150)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1)
    mode = "FULL SWEEP" if result.full_sweep else "ROI"
    tag = f"{mode} | {pipeline.drive_side.value.upper()}"
    cv2.putText(frame, tag, (rx1, max(14, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _annotate(pipeline, frame, result: FrameResult, state: CabinState) -> None:
    """Draw box, landmarks, head-pose axis, gaze arrow, and the panel onto a frame."""
    _draw_roi(pipeline, frame, result)
    if result.bbox is not None:
        b = result.bbox
        color = _ALERT_COLORS.get(state.driver_state, WHITE)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        _draw_landmarks(pipeline, frame, b)
        cx, cy = int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2)
        if pipeline.headpose_estimator is not None:
            hp = result.head_pose
            pipeline.headpose_estimator.model.draw_axis(
                frame, hp.yaw, hp.pitch, hp.roll, tdx=cx, tdy=cy, size=int((b.x2 - b.x1) * 0.6)
            )
        _draw_gaze(frame, cx, cy, result.gaze_yaw, result.gaze_pitch)
        eyes = "CLOSED (blink)" if result.ear < EAR_CLOSED_THRESHOLD else "open"
        cv2.putText(frame, f"eyes: {eyes}", (int(b.x1), int(b.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        _draw_class_boxes(frame, b, state)
    _draw_detections(frame, result)
    _panel(frame, result, state)


def main() -> None:
    """Annotate a video with the full DMS pipeline and write the result."""
    parser = argparse.ArgumentParser(description="CabinSense DMS video demo")
    parser.add_argument("--input", default="data/videos/incabin_sample.mp4")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "dms_demo.mp4"))
    parser.add_argument("--drive-side", choices=[d.value for d in DriveSide], default=None,
                        help="lhd/rhd; omit to auto-detect from the driver's position")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        print(f"could not open {args.input}")
        return
    fps = int(capture.get(cv2.CAP_PROP_FPS)) or 24
    build_kwargs = (
        {"drive_side": DriveSide(args.drive_side), "autoinit_drive_side": False}
        if args.drive_side else {}
    )
    pipeline = build_pipeline(fps=fps, **build_kwargs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, OUT_SIZE)
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result, state = pipeline.step(frame)
        _annotate(pipeline, frame, result, state)
        writer.write(cv2.resize(frame, OUT_SIZE))
        frames += 1
    capture.release()
    writer.release()
    print(f"wrote {args.output} ({frames} frames)")


if __name__ == "__main__":
    main()
