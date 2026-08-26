"""Generic training loop with MLflow tracking and best-checkpoint saving.

Parameterized by a per-batch ``step_fn`` (returns the loss) and an ``eval_fn``
(returns a monitored scalar plus a metrics dict), so the same loop drives both the
single-label drowsiness model and the masked multi-task attribute model. Seeded and
reproducible; logs params/metrics to a local MLflow file store.
"""

from __future__ import annotations

import os

# Opt into MLflow's local file-store backend (deprecated-but-supported), so runs are
# tracked to a local directory without needing a database server.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, FACE_INPUT_SIZE, MLRUNS_DIR
from src.train.utils import set_seed

StepFn = Callable[[torch.nn.Module, Any, str], torch.Tensor]
EvalFn = Callable[[torch.nn.Module, DataLoader, str], tuple[float, dict[str, Any]]]


@dataclass
class TrainConfig:
    """Hyperparameters and run settings for a training job."""

    run_name: str
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_workers: int = 0            # 0 is safest on Windows
    img_size: int = FACE_INPUT_SIZE
    experiment: str = "cabinsense"


def get_device() -> str:
    """Return 'cuda' when available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(dataset: Any, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Build a DataLoader with project-standard settings."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )


def run_training(
    model: torch.nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    step_fn: StepFn,
    eval_fn: EvalFn,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Train ``model``, logging to MLflow and saving the best checkpoint."""
    set_seed()
    device = get_device()
    model.to(device)
    train_loader = make_loader(train_dataset, cfg.batch_size, True, cfg.num_workers)
    val_loader = make_loader(val_dataset, cfg.batch_size, False, cfg.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{cfg.run_name}.pt"
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(cfg.experiment)

    best_monitor = float("-inf")
    best_metrics: dict[str, Any] = {}
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {"epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size, "img_size": cfg.img_size}
        )
        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = step_fn(model, batch, device)
                loss.backward()
                optimizer.step()
                running += float(loss.item())
            monitor, metrics = eval_fn(model, val_loader, device)
            mlflow.log_metric("train_loss", running / max(1, len(train_loader)), step=epoch)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value), step=epoch)
            if monitor > best_monitor:
                best_monitor = monitor
                best_metrics = metrics
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_metric("best_monitor", best_monitor, step=epoch)
    return {"checkpoint": str(checkpoint_path), "best_monitor": best_monitor, "metrics": best_metrics}
