"""Verify PyTorch <-> ONNX output parity (Phase 6).

Confirms the exported ONNX models reproduce the PyTorch checkpoints within
tolerance, so the ONNX artifacts are the frozen source of truth any future SoC port
would match. Run after training and export.

Run from the project root:
    python scripts/verify_parity.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import CHECKPOINT_DIR, DEPLOY_MODELS_DIR, FACE_INPUT_SIZE
from src.data.datasets import DROWSINESS_CLASSES
from src.models.classifier import ImageClassifier
from src.models.multitask import MultiTaskClassifier
from src.runtime.onnx_session import OnnxModel

TOLERANCE = 1e-3


def _max_diff(model: torch.nn.Module, checkpoint: pathlib.Path, onnx_path: pathlib.Path) -> float:
    """Return the max absolute difference between torch and ONNX outputs."""
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE)
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = OnnxModel(onnx_path).run(dummy.numpy())[0]
    return float(np.max(np.abs(torch_out - onnx_out)))


def main() -> None:
    """Check parity for both models where checkpoints and ONNX exports exist."""
    checks = [
        (ImageClassifier(num_classes=len(DROWSINESS_CLASSES), pretrained=False), "drowsiness"),
        (MultiTaskClassifier(pretrained=False), "multitask"),
    ]
    all_ok = True
    for model, name in checks:
        checkpoint = CHECKPOINT_DIR / f"{name}.pt"
        onnx_path = DEPLOY_MODELS_DIR / f"{name}.onnx"
        if not (checkpoint.exists() and onnx_path.exists()):
            print(f"skip {name}: missing checkpoint or ONNX export")
            continue
        diff = _max_diff(model, checkpoint, onnx_path)
        ok = diff < TOLERANCE
        all_ok = all_ok and ok
        print(f"{name}: max_abs_diff={diff:.2e} {'OK' if ok else 'FAIL'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
