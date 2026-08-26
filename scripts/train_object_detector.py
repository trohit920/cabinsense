"""Fine-tune a YOLOv8n consolidated detector for smoking / seatbelt / cap.

Trains one multi-class detector on the merged dataset from
``scripts/prepare_unified_detection.py`` so a single inference serves all three
object-like classes COCO does not cover (SoC-friendly). Validates, reports per-class
AP, and copies the best checkpoint to ``deploy/models/custom_yolo.pt``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CUSTOM_DETECTOR_MODEL,
    DATA_DIR,
    DEPLOY_MODELS_DIR,
    OUTPUT_DIR,
    SEED,
)

DATA_YAML = DATA_DIR / "detect_custom" / "data.yaml"
BASE_MODEL = "yolov8n.pt"          # COCO-pretrained starting point
RUN_DIR = OUTPUT_DIR / "detect"
RUN_NAME = "custom"
DEPLOY_WEIGHTS = DEPLOY_MODELS_DIR / CUSTOM_DETECTOR_MODEL


def _parse_args() -> argparse.Namespace:
    """CLI knobs for a full run or a quick smoke run."""
    parser = argparse.ArgumentParser(description="Train the consolidated custom detector")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)     # fits a 6 GB GPU
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="0")             # "0" GPU, "cpu" fallback
    return parser.parse_args()


def main() -> None:
    """Train, validate, and deploy the consolidated detector."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} missing - run scripts/prepare_unified_detection.py")
    from ultralytics import YOLO

    args = _parse_args()
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=SEED,
        project=str(RUN_DIR),
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
    )
    metrics = model.val(data=str(DATA_YAML), device=args.device, verbose=False)
    print(f"mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")
    for i, name in metrics.names.items():
        print(f"  {name:12s} AP50 {metrics.box.ap50[i]:.3f}")

    best = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    DEPLOY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, DEPLOY_WEIGHTS)
    print(f"deployed {best} -> {DEPLOY_WEIGHTS}")


if __name__ == "__main__":
    main()
