"""Train the multi-task attribute model (Phase 3).

Trains a shared-backbone multi-task classifier with masked BCE on CelebA
(glasses, cap); other tasks (seatbelt/phone/smoking/yawn) are masked out until
labeled data is added. Tracked with MLflow, monitored by mean F1 over labeled
tasks. Downloads CelebA on first run.

Run from the project root:
    python scripts/train_multitask.py --epochs 6 --batch-size 64
    python scripts/train_multitask.py --max-samples 6000   # quick smoke run
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import ATTRIBUTE_TASKS
from src.data.datasets import build_celeba_multitask_datasets
from src.eval.metrics import masked_multitask_metrics
from src.models.multitask import MultiTaskClassifier
from src.train.losses import masked_bce_with_logits
from src.train.trainer import TrainConfig, run_training


def step_fn(model: torch.nn.Module, batch, device: str) -> torch.Tensor:
    """Masked BCE training step for the multi-task heads."""
    images, labels, mask = batch
    logits = model(images.to(device))
    return masked_bce_with_logits(logits, labels.to(device), mask.to(device))


def eval_fn(model: torch.nn.Module, loader, device: str) -> tuple[float, dict]:
    """Evaluate per-task masked accuracy/F1; monitor = mean F1 over labeled tasks."""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for images, labels, mask in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            preds.append((probs >= 0.5).astype(np.float32))
            trues.append(labels.numpy())
            masks.append(mask.numpy())
    per_task = masked_multitask_metrics(
        np.concatenate(trues), np.concatenate(preds), np.concatenate(masks), ATTRIBUTE_TASKS
    )
    f1s = [v["f1"] for v in per_task.values() if v["support"] > 0]
    monitor = float(np.mean(f1s)) if f1s else 0.0
    flat = {f"{task}_f1": v["f1"] for task, v in per_task.items() if v["support"] > 0}
    flat["mean_f1"] = monitor
    return monitor, flat


def main() -> None:
    """Parse arguments and train the multi-task attribute model."""
    parser = argparse.ArgumentParser(description="Train the multi-task attribute model")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=20, help="CelebA parquet shards to use")
    args = parser.parse_args()

    datasets = build_celeba_multitask_datasets(max_samples=args.max_samples, num_shards=args.num_shards)
    model = MultiTaskClassifier()
    cfg = TrainConfig(run_name="multitask", epochs=args.epochs, batch_size=args.batch_size)
    result = run_training(model, datasets["train"], datasets["val"], step_fn, eval_fn, cfg)
    print(result)


if __name__ == "__main__":
    main()
