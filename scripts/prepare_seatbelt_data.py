"""Prepare the seatbelt YOLO detection dataset for training.

Downloads the Roboflow-exported seatbelt package (CC BY 4.0) from the Hugging Face
Hub, extracts the train/valid/test image+label splits out of its nested folders into
a clean ``data/seatbelt_yolo`` layout, and writes a portable ``data.yaml``. Idempotent:
re-running with the target present is a no-op. Run from the project root.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from src.config import DATA_DIR

REPO_ID = "ramankamran/seatbelt-detection-v2i-yolov11-lt"
ZIP_NAME = "seatbelt_detection_v2i_yolov11_lt_package.zip"
# Root of the Roboflow export inside the zip (doubly nested, with spaces).
INNER_ROOT = "Seatbelt Detection.v2i.yolov11/Seatbelt Detection.v2i.yolov11"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("person-noseatbelt", "person-seatbelt")
TARGET_DIR = DATA_DIR / "seatbelt_yolo"


def _extract_split(archive: zipfile.ZipFile, split: str, dest: Path) -> int:
    """Copy one split's images and labels out of the archive into ``dest``."""
    (dest / split / "images").mkdir(parents=True, exist_ok=True)
    (dest / split / "labels").mkdir(parents=True, exist_ok=True)
    count = 0
    prefix = f"{INNER_ROOT}/{split}/"
    for name in archive.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        kind = "images" if "/images/" in name else "labels" if "/labels/" in name else None
        if kind is None:
            continue
        with archive.open(name) as src:
            (dest / split / kind / Path(name).name).write_bytes(src.read())
            count += 1 if kind == "images" else 0
    return count


def _write_yaml(dest: Path) -> None:
    """Write a YOLO data.yaml with absolute split paths and class names."""
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    dest.joinpath("data.yaml").write_text(
        f"path: {dest.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )


def main() -> None:
    """Download, extract, and lay out the seatbelt detection dataset."""
    if (TARGET_DIR / "data.yaml").exists():
        print(f"seatbelt dataset already prepared at {TARGET_DIR}")
        return
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    zip_path = hf_hub_download(REPO_ID, ZIP_NAME, repo_type="dataset")
    with zipfile.ZipFile(zip_path) as archive:
        for split in SPLITS:
            n_images = _extract_split(archive, split, TARGET_DIR)
            print(f"{split}: {n_images} images")
    _write_yaml(TARGET_DIR)
    print(f"wrote {TARGET_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
