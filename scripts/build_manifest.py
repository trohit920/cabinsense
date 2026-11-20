"""Build the deployment manifest (Phase 7).

Writes deploy/model_manifest.json - the single contract a future C/C++ SoC port
consumes: per-model input spec and normalization for the learned classifiers, the
detector/landmark artifacts, and all decision thresholds and temporal parameters.
Validated with src.core.manifest before writing.

Run from the project root:
    python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.core.manifest import validate_manifest
from src.data.datasets import DROWSINESS_CLASSES


def _classifier_spec(file: str, extra: dict) -> dict:
    """Common input/normalization spec for a face-crop classifier."""
    size = config.FACE_INPUT_SIZE
    spec = {
        "file": file,
        "input_name": "input",
        "input_shape": [1, 3, size, size],
        "layout": "NCHW",
        "color": "RGB",
        "mean": list(config.IMAGENET_MEAN),
        "std": list(config.IMAGENET_STD),
        "output_names": ["output"],
    }
    spec.update(extra)
    return spec


def build_manifest() -> dict:
    """Assemble the manifest dictionary from project config."""
    return {
        "models": {
            "drowsiness": _classifier_spec(
                "drowsiness.onnx", {"classes": list(DROWSINESS_CLASSES), "activation": "softmax"}
            ),
            "multitask": _classifier_spec(
                "multitask.onnx",
                {"tasks": list(config.ATTRIBUTE_TASKS), "activation": "sigmoid", "threshold": 0.5},
            ),
        },
        "detectors": {
            "face": {"file": "yunet.onnx", "type": "yunet", "input_size": list(config.DETECT_INPUT_SIZE)},
            "landmarks": {"file": "pfld.onnx", "type": "pfld", "num_points": 98, "input_size": 112},
        },
        "thresholds": {
            "ear_closed": config.EAR_CLOSED_THRESHOLD,
            "mar_yawn": config.MAR_YAWN_THRESHOLD,
            "perclos_drowsy_ratio": config.PERCLOS_DROWSY_RATIO,
            "distraction_off_road_ratio": config.DISTRACTION_OFF_ROAD_RATIO,
            "min_confidence": config.MIN_CONFIDENCE,
        },
        "temporal": {
            "perclos_window_s": config.PERCLOS_WINDOW_SECONDS,
            "distraction_window_s": config.DISTRACTION_WINDOW_SECONDS,
            "smoothing_window_s": config.SMOOTHING_WINDOW_SECONDS,
            "hysteresis_on_ratio": config.HYSTERESIS_ON_RATIO,
            "hysteresis_off_ratio": config.HYSTERESIS_OFF_RATIO,
        },
    }


def main() -> None:
    """Validate and write the deployment manifest."""
    manifest = build_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit(f"manifest invalid: {errors}")
    config.DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DEPLOY_DIR / "model_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
