"""Dataset loaders for the confirmed public HuggingFace datasets.

Reads the parquet shards directly via huggingface_hub + pandas instead of
``datasets.load_dataset``, which on Windows locks its arrow cache during the
"prepare" step (WinError 32). Wraps the frames in a common torch Dataset interface
with seeded, leakage-free splits and optional subsampling/shard limits. The
multi-task dataset returns per-task labels plus a validity mask, so a sample from a
dataset that only labels some tasks contributes only to those.
"""

from __future__ import annotations

import os

# Disable the hf_xet download backend, which holds file handles on Windows and
# causes WinError 32 during cache cleanup; the classic path is robust.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import io
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from torch.utils.data import Dataset

from src.config import ATTRIBUTE_TASKS, FACE_INPUT_SIZE, SEED
from src.data.transforms import build_eval_transforms, build_train_transforms

DATASET_DROWSINESS: str = "akahana/Driver-Drowsiness-Dataset"
DATASET_CELEBA: str = "huggan/CelebA-faces-with-attributes"

DROWSINESS_CLASSES: tuple[str, ...] = ("Drowsy", "Non Drowsy")  # ClassLabel order (0, 1)
CELEBA_TASK_ATTRIBUTES: dict[str, str] = {"glasses": "Eyeglasses", "cap": "Wearing_Hat"}
_CELEBA_DEFAULT_SHARDS: int = 20  # ~30k images; bounds the download for a POC


def seeded_split(
    n: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    """Deterministically partition indices [0, n) into train/val/test."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1.0")
    order = np.random.RandomState(SEED).permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def _shard_paths(repo_id: str, split: str, num_shards: int | None) -> list[str]:
    """Download the parquet shards for a split and return local file paths."""
    files = sorted(
        f
        for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(f"data/{split}-") and f.endswith(".parquet")
    )
    if num_shards is not None:
        files = files[:num_shards]
    return [hf_hub_download(repo_id, f, repo_type="dataset") for f in files]


def _read_frame(repo_id: str, split: str, num_shards: int | None) -> pd.DataFrame:
    """Load a concatenated DataFrame from a split's parquet shards."""
    paths = _shard_paths(repo_id, split, num_shards)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _decode_image(cell: object) -> Image.Image:
    """Decode an HF parquet image cell (dict with bytes) into an RGB PIL image."""
    data = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(data)).convert("RGB")


def _drowsy_label(row: pd.Series) -> int:
    """Extract the integer drowsiness label from a row."""
    return int(row["label"])


class SingleLabelDataset(Dataset):
    """Wraps a DataFrame of images with a single integer label per row."""

    def __init__(self, frame: pd.DataFrame, label_fn: Callable[[pd.Series], int], transform: Callable) -> None:
        """Bind a frame, a label-extraction function, and an image transform."""
        self._frame = frame.reset_index(drop=True)
        self._label_fn = label_fn
        self._transform = transform

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int):
        row = self._frame.iloc[index]
        return self._transform(_decode_image(row["image"])), self._label_fn(row)


class MultiTaskAttributeDataset(Dataset):
    """Wraps a DataFrame with per-task binary labels and a validity mask."""

    def __init__(
        self,
        frame: pd.DataFrame,
        task_attributes: dict[str, str],
        transform: Callable,
        tasks: tuple[str, ...] = ATTRIBUTE_TASKS,
    ) -> None:
        """Bind a frame and the map of {task -> attribute column} it supplies."""
        self._frame = frame.reset_index(drop=True)
        self._task_attributes = task_attributes
        self._transform = transform
        self._tasks = tasks

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int):
        row = self._frame.iloc[index]
        image = self._transform(_decode_image(row["image"]))
        labels = torch.zeros(len(self._tasks), dtype=torch.float32)
        mask = torch.zeros(len(self._tasks), dtype=torch.float32)
        for position, task in enumerate(self._tasks):
            attribute = self._task_attributes.get(task)
            if attribute is not None:
                labels[position] = 1.0 if row[attribute] > 0 else 0.0
                mask[position] = 1.0
        return image, labels, mask


def _cap(indices: np.ndarray, max_samples: int | None) -> np.ndarray:
    """Optionally truncate an index array to at most ``max_samples`` entries."""
    return indices[:max_samples] if max_samples is not None else indices


def build_drowsiness_datasets(
    max_samples: int | None = None, img_size: int = FACE_INPUT_SIZE, num_shards: int | None = None
) -> dict[str, SingleLabelDataset]:
    """Build train/val (from train shards) and test (from test shards) datasets."""
    train_frame = _read_frame(DATASET_DROWSINESS, "train", num_shards)
    test_frame = _read_frame(DATASET_DROWSINESS, "test", None)
    split = seeded_split(len(train_frame), (0.9, 0.1, 0.0))
    train_tf = build_train_transforms(img_size)
    eval_tf = build_eval_transforms(img_size)
    return {
        "train": SingleLabelDataset(train_frame.iloc[_cap(split["train"], max_samples)], _drowsy_label, train_tf),
        "val": SingleLabelDataset(train_frame.iloc[_cap(split["val"], max_samples)], _drowsy_label, eval_tf),
        "test": SingleLabelDataset(test_frame.iloc[_cap(np.arange(len(test_frame)), max_samples)], _drowsy_label, eval_tf),
    }


def build_drowsiness_test(
    transform: Callable, max_samples: int | None = None
) -> SingleLabelDataset:
    """Build the drowsiness test split with a caller-supplied transform."""
    test_frame = _read_frame(DATASET_DROWSINESS, "test", None)
    indices = _cap(np.arange(len(test_frame)), max_samples)
    return SingleLabelDataset(test_frame.iloc[indices], _drowsy_label, transform)


def build_celeba_multitask_datasets(
    max_samples: int | None = None,
    img_size: int = FACE_INPUT_SIZE,
    num_shards: int | None = _CELEBA_DEFAULT_SHARDS,
) -> dict[str, MultiTaskAttributeDataset]:
    """Build seeded train/val/test multi-task datasets (glasses/cap) from CelebA."""
    frame = _read_frame(DATASET_CELEBA, "train", num_shards)
    split = seeded_split(len(frame))
    train_tf = build_train_transforms(img_size)
    eval_tf = build_eval_transforms(img_size)
    out: dict[str, MultiTaskAttributeDataset] = {}
    for name in ("train", "val", "test"):
        transform = train_tf if name == "train" else eval_tf
        subset = frame.iloc[_cap(split[name], max_samples)]
        out[name] = MultiTaskAttributeDataset(subset, CELEBA_TASK_ATTRIBUTES, transform)
    return out
