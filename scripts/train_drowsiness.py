"""Train the binary drowsiness classifier (Phase 2).

Fine-tunes a MobileNetV3 classifier on akahana/Driver-Drowsiness-Dataset, tracked
with MLflow and monitored by macro-F1. Downloads the dataset on first run.

Run from the project root:
    python scripts/train_drowsiness.py --epochs 8 --batch-size 64
    python scripts/train_drowsiness.py --max-samples 4000   # quick smoke run
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.data.datasets import DROWSINESS_CLASSES, build_drowsiness_datasets
from src.eval.metrics import classification_metrics
from src.models.classifier import ImageClassifier
from src.train.trainer import TrainConfig, run_training


def step_fn(model: torch.nn.Module, batch, device: str) -> torch.Tensor:
    """Cross-entropy training step for single-label drowsiness."""
    images, labels = batch
    logits = model(images.to(device))
    return F.cross_entropy(logits, labels.to(device))


def eval_fn(model: torch.nn.Module, loader, device: str) -> tuple[float, dict]:
    """Evaluate accuracy/macro-F1 on the validation loader."""
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            trues.extend(labels.tolist())
    metrics = classification_metrics(trues, preds)
    return metrics["f1_macro"], metrics


def main() -> None:
    """Parse arguments and train the drowsiness classifier."""
    parser = argparse.ArgumentParser(description="Train the drowsiness classifier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None, help="limit train parquet shards")
    args = parser.parse_args()

    datasets = build_drowsiness_datasets(max_samples=args.max_samples, num_shards=args.num_shards)
    model = ImageClassifier(num_classes=len(DROWSINESS_CLASSES))
    cfg = TrainConfig(run_name="drowsiness", epochs=args.epochs, batch_size=args.batch_size)
    result = run_training(model, datasets["train"], datasets["val"], step_fn, eval_fn, cfg)
    print(result)


if __name__ == "__main__":
    main()
