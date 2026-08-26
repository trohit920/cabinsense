"""End-to-end CabinSense inference pipeline (system integration).

Ties detection, landmarks, geometry (EAR/MAR/pose/gaze), and the learned models
into a per-frame FrameResult, then applies temporal fusion (PERCLOS, distraction
window, hysteresis flags, majority-vote gaze/occupancy) to produce a stable
CabinState. Learned models are optional so the geometry-only pipeline runs before
any training. Uses the portable core/ for all decision logic.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from src.config import (
    ATTRIBUTE_CONFIDENCE_THRESHOLD,
    ATTRIBUTE_TASKS,
    CLASSIFIER_ATTRIBUTES,
    DEFAULT_DRIVE_SIDE,
    DETECTOR_ATTRIBUTES,
    DISTRACTION_OFF_ROAD_RATIO,
    DISTRACTION_WINDOW_SECONDS,
    DRIVE_SIDE_AUTOINIT_SAMPLES,
    DROWSY_MODEL_CLOSED_PROB,
    EAR_CLOSED_THRESHOLD,
    FACE_INPUT_SIZE,
    HYSTERESIS_OFF_RATIO,
    HYSTERESIS_ON_RATIO,
    MAR_YAWN_THRESHOLD,
    PERCLOS_WINDOW_SECONDS,
    ROI_FULL_SWEEP_SECONDS,
    SMOOTHING_WINDOW_SECONDS,
    DriveSide,
)
from src.core.features import eye_aspect_ratio, mouth_aspect_ratio
from src.core.fusion import fuse_driver_state, gaze_off_road, is_distracted, is_drowsy
from src.core.gaze import head_pose_to_zone
from src.core.landmark_indices import (
    left_eye_points,
    mouth_points,
    pose_points,
    right_eye_points,
)
from src.core.pose import estimate_head_pose
from src.core.preprocess import preprocess_crop
from src.core.roi import compute_driver_roi, infer_drive_side, offset_box, select_driver
from src.core.temporal import HysteresisFlag, RatioWindow, majority_label
from src.runtime.detect import FaceDetector
from src.runtime.landmarks import LandmarkDetector
from src.runtime.onnx_session import OnnxModel
from src.schemas import BBox, CabinState, FrameResult, HeadPose, ObjectDetection


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable 1D softmax."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _window(seconds: float, fps: int) -> int:
    """Convert a duration in seconds to a frame count (at least 1)."""
    return max(1, int(seconds * fps))


class CabinSensePipeline:
    """Stateful multi-task in-cabin monitoring pipeline."""

    def __init__(
        self,
        detector: FaceDetector,
        landmarker: LandmarkDetector,
        drowsiness_model: OnnxModel | None = None,
        multitask_model: OnnxModel | None = None,
        headpose_estimator: object | None = None,
        gaze_estimator: object | None = None,
        object_detectors: list | None = None,
        fps: int = 30,
        drive_side: DriveSide = DEFAULT_DRIVE_SIDE,
        autoinit_drive_side: bool = True,
    ) -> None:
        """Bind models/estimators and initialize temporal state for the stream FPS."""
        self.detector = detector
        self.landmarker = landmarker
        self.drowsiness_model = drowsiness_model
        self.multitask_model = multitask_model
        self.headpose_estimator = headpose_estimator  # 6DRepNet (falls back to PnP)
        self.gaze_estimator = gaze_estimator          # L2CS (optional)
        # Localized detectors (COCO phone/drinking + custom seatbelt); each maps its
        # classes to CabinSense attribute labels.
        self.object_detectors = list(object_detectors or [])
        # Driver ROI + two-rate scheduling: process an ROI each frame, sweep the full
        # frame every ROI_FULL_SWEEP_SECONDS to recount occupants / re-localize.
        self.drive_side = drive_side
        self._full_sweep_every = _window(ROI_FULL_SWEEP_SECONDS, fps)
        self._frame_index = -1
        self._driver_roi: tuple[int, int, int, int] | None = None
        self._occupant_count = 0
        self._drive_side_locked = not autoinit_drive_side
        self._drive_side_votes: list[DriveSide] = []
        self._perclos = RatioWindow(_window(PERCLOS_WINDOW_SECONDS, fps))
        # The learned eye-state classifier gets its own window over the same horizon,
        # so a single high-probability frame cannot latch the drowsiness flag.
        self._model_perclos = RatioWindow(_window(PERCLOS_WINDOW_SECONDS, fps))
        self._distraction = RatioWindow(_window(DISTRACTION_WINDOW_SECONDS, fps))
        smoothing = _window(SMOOTHING_WINDOW_SECONDS, fps)
        self._gaze_history: deque[str] = deque(maxlen=smoothing)
        self._count_history: deque[int] = deque(maxlen=smoothing)
        self._flags = {
            task: HysteresisFlag(smoothing, HYSTERESIS_ON_RATIO, HYSTERESIS_OFF_RATIO)
            for task in ATTRIBUTE_TASKS
        }

    def _drowsiness_prob(self, crop_bgr: np.ndarray) -> float:
        """Learned P(drowsy) for a face crop (0.0 when no model is loaded)."""
        if self.drowsiness_model is None:
            return 0.0
        logits = self.drowsiness_model.run(preprocess_crop(crop_bgr, FACE_INPUT_SIZE))[0]
        return float(_softmax(logits.reshape(-1))[0])  # index 0 = "Drowsy"

    def _attribute_probs(self, crop_bgr: np.ndarray) -> dict[str, float]:
        """Learned per-task attribute probabilities (empty when no model)."""
        if self.multitask_model is None:
            return {}
        logits = self.multitask_model.run(preprocess_crop(crop_bgr, FACE_INPUT_SIZE))[0].reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return {task: float(probs[i]) for i, task in enumerate(ATTRIBUTE_TASKS)}

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """Locate the driver (ROI or full sweep) and run the per-driver models."""
        self._frame_index += 1
        full_sweep = self._driver_roi is None or self._frame_index % self._full_sweep_every == 0
        box, roi = self._locate_driver(frame_bgr, full_sweep)
        if box is None:
            return FrameResult(occupant_count=0, full_sweep=full_sweep)
        return self._process_driver(frame_bgr, box, roi, full_sweep)

    def _locate_driver(
        self, frame_bgr: np.ndarray, full_sweep: bool
    ) -> tuple[BBox | None, tuple[int, int, int, int] | None]:
        """Find the driver via a full-frame sweep or a cheaper ROI-only detection."""
        frame_w, frame_h = frame_bgr.shape[1], frame_bgr.shape[0]
        if full_sweep:
            boxes = self.detector.detect(frame_bgr)
            self._occupant_count = len(boxes)
            self._maybe_lock_drive_side(boxes, frame_w)
        else:
            rx1, ry1, rx2, ry2 = self._driver_roi
            crop = frame_bgr[ry1:ry2, rx1:rx2]
            boxes = [offset_box(b, rx1, ry1) for b in self.detector.detect(crop)] if crop.size else []
        box = select_driver(boxes, frame_w, self.drive_side)
        if box is None:
            self._driver_roi = None
            return None, None
        self._driver_roi = compute_driver_roi(box, frame_w, frame_h)
        return box, self._driver_roi

    def _maybe_lock_drive_side(self, boxes: list[BBox], frame_w: int) -> None:
        """Vote the steering side from the largest face until locked (auto-init)."""
        if self._drive_side_locked or not boxes:
            return
        largest = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
        self._drive_side_votes.append(infer_drive_side((largest.x1 + largest.x2) / 2.0, frame_w))
        if len(self._drive_side_votes) >= DRIVE_SIDE_AUTOINIT_SAMPLES:
            self.drive_side = Counter(self._drive_side_votes).most_common(1)[0][0]
            self._drive_side_locked = True

    def _process_driver(
        self,
        frame_bgr: np.ndarray,
        box: BBox,
        roi: tuple[int, int, int, int] | None,
        full_sweep: bool,
    ) -> FrameResult:
        """Run landmarks, geometry, pose, gaze, and learned models for the driver."""
        landmarks = self.landmarker.predict(frame_bgr, box)
        ear = 0.5 * (eye_aspect_ratio(left_eye_points(landmarks))
                     + eye_aspect_ratio(right_eye_points(landmarks)))
        mar = mouth_aspect_ratio(mouth_points(landmarks))
        face_crop = frame_bgr[max(0, int(box.y1)):int(box.y2), max(0, int(box.x1)):int(box.x2)]
        attr_crop = self._padded_crop(frame_bgr, box)  # includes hair/hat region
        pose = self._head_pose(face_crop, landmarks, frame_bgr.shape)
        gaze_yaw, gaze_pitch = self._gaze(face_crop)
        mirror = self.drive_side == DriveSide.RIGHT_HAND
        return FrameResult(
            bbox=box,
            ear=ear,
            mar=mar,
            head_pose=pose,
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
            eye_closed_prob=self._drowsiness_prob(face_crop) if face_crop.size else 0.0,
            attribute_probs=self._attribute_probs(attr_crop) if attr_crop.size else {},
            object_detections=self._detect_objects(frame_bgr, roi),
            gaze_zone=head_pose_to_zone(pose.yaw, pose.pitch, mirror=mirror),
            occupant_count=self._occupant_count,
            driver_roi=roi,
            full_sweep=full_sweep,
        )

    def _detect_objects(
        self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None
    ) -> list:
        """Run localized detectors on the driver ROI crop, boxes mapped to full frame."""
        if not self.object_detectors:
            return []
        if roi is None:
            crop, off_x, off_y = frame_bgr, 0, 0
        else:
            rx1, ry1, rx2, ry2 = roi
            crop, off_x, off_y = frame_bgr[ry1:ry2, rx1:rx2], rx1, ry1
        if not crop.size:
            return []
        detections: list = []
        for detector in self.object_detectors:
            for det in detector.detect(crop):
                detections.append(ObjectDetection(det.label, offset_box(det.box, off_x, off_y)))
        return detections

    def _head_pose(self, face_crop: np.ndarray, landmarks: np.ndarray, shape: tuple) -> HeadPose:
        """Head pose from 6DRepNet when available, else solvePnP on landmarks."""
        if self.headpose_estimator is not None and face_crop.size:
            return self.headpose_estimator.predict(face_crop)
        height, width = shape[:2]
        return estimate_head_pose(pose_points(landmarks), (width, height))

    def _gaze(self, face_crop: np.ndarray) -> tuple[float, float]:
        """Gaze (yaw, pitch) in degrees from L2CS when available, else zeros."""
        if self.gaze_estimator is not None and face_crop.size:
            return self.gaze_estimator.predict(face_crop)
        return 0.0, 0.0

    @staticmethod
    def _padded_crop(frame_bgr: np.ndarray, box: BBox) -> np.ndarray:
        """Crop the face box padded outward (esp. upward) to include hair/hat.

        The attribute model was trained on full aligned face images, so it needs the
        hat/hair region a tight detector box would exclude (e.g. caps on the head).
        """
        height, width = frame_bgr.shape[:2]
        box_w, box_h = box.x2 - box.x1, box.y2 - box.y1
        x1 = max(0, int(box.x1 - 0.20 * box_w))
        x2 = min(width, int(box.x2 + 0.20 * box_w))
        y1 = max(0, int(box.y1 - 0.60 * box_h))
        y2 = min(height, int(box.y2 + 0.10 * box_h))
        return frame_bgr[y1:y2, x1:x2]

    def update(self, frame_bgr: np.ndarray) -> CabinState:
        """Process a frame and fold it into the temporally-smoothed CabinState."""
        return self._fuse(self.process_frame(frame_bgr))

    def step(self, frame_bgr: np.ndarray) -> tuple[FrameResult, CabinState]:
        """Process a frame once and return both the raw result and fused state."""
        frame = self.process_frame(frame_bgr)
        return frame, self._fuse(frame)

    def _fuse(self, frame: FrameResult) -> CabinState:
        """Fold a per-frame result into the temporally-smoothed CabinState."""
        self._count_history.append(frame.occupant_count)
        occupants = Counter(self._count_history).most_common(1)[0][0]
        if frame.occupant_count == 0:
            return CabinState(occupant_count=occupants)

        # EAR and the drowsiness model are only reliable near-frontal; when the head
        # is turned off-road, that is distraction, not drowsiness.
        frontal = not gaze_off_road(frame.gaze_zone)
        perclos_value = self._perclos.update(frontal and frame.ear < EAR_CLOSED_THRESHOLD)
        model_closed = frame.eye_closed_prob >= DROWSY_MODEL_CLOSED_PROB
        model_perclos = self._model_perclos.update(frontal and model_closed)
        off_road_ratio = self._distraction.update(gaze_off_road(frame.gaze_zone))
        # Both eye-closure signals are accumulated over the PERCLOS window before they
        # can raise the flag. Thresholding the classifier per-frame latched DROWSY on a
        # single frame at p=0.53 with EAR 0.30 and PERCLOS 0.00 - a blink, not drowsiness.
        drowsy = frontal and (is_drowsy(perclos_value) or is_drowsy(model_perclos))
        distracted = is_distracted(off_road_ratio, DISTRACTION_OFF_ROAD_RATIO)
        self._gaze_history.append(frame.gaze_zone)

        flags = self._update_flags(frame)
        return CabinState(
            driver_state=fuse_driver_state(drowsy, distracted),
            gaze_zone=majority_label(list(self._gaze_history)),
            yawning=flags["yawn"],
            phone=flags["phone"],
            smoking=flags["smoking"],
            seatbelt=flags["seatbelt"],
            glasses=flags["glasses"],
            cap=flags["cap"],
            drinking=flags["drinking"],
            occupant_count=occupants,
            confidences={
                "perclos": perclos_value,
                "model_perclos": model_perclos,
                "off_road": off_road_ratio,
                "drowsy_prob": frame.eye_closed_prob,
            },
        )

    def _update_flags(self, frame: FrameResult) -> dict[str, bool]:
        """Update per-attribute hysteresis flags from this frame's trusted signals.

        Signal precedence per task: yawn from MAR geometry; object-like classes from a
        localized detector (phone, drinking via COCO YOLO; smoking, seatbelt, cap via the
        consolidated custom YOLO); genuine face attributes (glasses) from the whole-image
        classifier at a high threshold; anything else reports False rather than emitting
        confident classifier hallucinations on out-of-domain cabin frames.
        """
        detected = {det.label for det in frame.object_detections}
        states: dict[str, bool] = {}
        for task in ATTRIBUTE_TASKS:
            if task == "yawn":
                active = frame.mar > MAR_YAWN_THRESHOLD
            elif task in DETECTOR_ATTRIBUTES:
                active = task in detected
            elif task in CLASSIFIER_ATTRIBUTES:
                active = frame.attribute_probs.get(task, 0.0) >= ATTRIBUTE_CONFIDENCE_THRESHOLD
            else:
                active = False
            states[task] = self._flags[task].update(active)
        return states
