"""Export trained checkpoints to ONNX and INT8 (Phase 6).

Loads the drowsiness and multi-task checkpoints, exports each to ONNX with static
shapes, and writes INT8-quantized variants into deploy/models/. Run after training.

Run from the project root:
    python scripts/export_and_quantize.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import CHECKPOINT_DIR, DEPLOY_MODELS_DIR, FACE_INPUT_SIZE
from src.data.datasets import DROWSINESS_CLASSES
from src.models.classifier import ImageClassifier
from src.models.multitask import MultiTaskClassifier
from src.optimize.export_onnx import export_model
from src.optimize.quantize import quantize_int8


def _export_one(model: torch.nn.Module, checkpoint: pathlib.Path, name: str) -> None:
    """Load a checkpoint into a model and write ONNX + INT8 variants."""
    if not checkpoint.exists():
        print(f"skip {name}: checkpoint not found at {checkpoint}")
        return
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    dummy = torch.randn(1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE)
    onnx_path = export_model(model, dummy, DEPLOY_MODELS_DIR / f"{name}.onnx")
    quantize_int8(onnx_path, DEPLOY_MODELS_DIR / f"{name}.int8.onnx")
    print(f"exported {name}: {onnx_path} (+ INT8)")


def main() -> None:
    """Export both trained models to ONNX + INT8."""
    _export_one(
        ImageClassifier(num_classes=len(DROWSINESS_CLASSES), pretrained=False),
        CHECKPOINT_DIR / "drowsiness.pt",
        "drowsiness",
    )
    _export_one(MultiTaskClassifier(pretrained=False), CHECKPOINT_DIR / "multitask.pt", "multitask")


if __name__ == "__main__":
    main()
