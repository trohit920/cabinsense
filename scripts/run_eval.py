"""Evaluate trained models: accuracy/F1, robustness, and failure analysis (Phase 4/5).

Scores the drowsiness model on the clean test split and under degraded (augmented)
conditions to quantify robustness, saves a confusion-matrix figure for failure
analysis, and reports masked per-task metrics for the multi-task model. Writes
outputs/eval.json. Run after training.

Run from the project root:
    python scripts/run_eval.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.config import (
    ATTRIBUTE_TASKS,
    CHECKPOINT_DIR,
    FACE_INPUT_SIZE,
    FIGURE_DIR,
    OUTPUT_DIR,
)
from src.data.datasets import (
    DROWSINESS_CLASSES,
    build_celeba_multitask_datasets,
    build_drowsiness_datasets,
    build_drowsiness_test,
)
from src.data.transforms import build_train_transforms
from src.eval.metrics import classification_metrics, masked_multitask_metrics
from src.models.classifier import ImageClassifier
from src.models.multitask import MultiTaskClassifier
from src.train.utils import set_seed


def _predict_single(model: torch.nn.Module, dataset, device: str) -> tuple[list[int], list[int]]:
    """Return (true, pred) label lists for a single-label dataset."""
    loader = DataLoader(dataset, batch_size=64)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
            trues.extend(labels.tolist())
    return trues, preds


def _save_confusion(trues: list[int], preds: list[int]) -> pathlib.Path:
    """Save a confusion-matrix figure for the drowsiness model."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(DROWSINESS_CLASSES)), DROWSINESS_CLASSES, rotation=20)
    ax.set_yticks(range(len(DROWSINESS_CLASSES)), DROWSINESS_CLASSES)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Drowsiness confusion matrix")
    fig.tight_layout()
    path = FIGURE_DIR / "confusion_drowsiness.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _eval_drowsiness(device: str) -> dict:
    """Clean + degraded (robustness) drowsiness metrics with a confusion figure."""
    checkpoint = CHECKPOINT_DIR / "drowsiness.pt"
    if not checkpoint.exists():
        return {"error": "drowsiness checkpoint not found"}
    model = ImageClassifier(num_classes=len(DROWSINESS_CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    clean = build_drowsiness_datasets()["test"]
    trues, preds = _predict_single(model, clean, device)
    _save_confusion(trues, preds)
    set_seed()  # seed the random augmentation for a reproducible robustness pass
    degraded = build_drowsiness_test(build_train_transforms(FACE_INPUT_SIZE))
    d_trues, d_preds = _predict_single(model, degraded, device)
    return {
        "clean": classification_metrics(trues, preds),
        "degraded": classification_metrics(d_trues, d_preds),
    }


def _eval_multitask(device: str) -> dict:
    """Masked per-task metrics for the multi-task model on its test split."""
    checkpoint = CHECKPOINT_DIR / "multitask.pt"
    if not checkpoint.exists():
        return {"error": "multitask checkpoint not found"}
    model = MultiTaskClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    loader = DataLoader(build_celeba_multitask_datasets()["test"], batch_size=64)
    preds, trues, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels, mask in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            preds.append((probs >= 0.5).astype(np.float32))
            trues.append(labels.numpy())
            masks.append(mask.numpy())
    return masked_multitask_metrics(
        np.concatenate(trues), np.concatenate(preds), np.concatenate(masks), ATTRIBUTE_TASKS
    )


def main() -> None:
    """Run drowsiness + multi-task evaluation and persist outputs/eval.json."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"drowsiness": _eval_drowsiness(device), "multitask": _eval_multitask(device)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
