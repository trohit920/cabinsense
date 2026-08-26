"""Merge the seatbelt, smoking, and cap detection datasets into one YOLO dataset.

Ingests the prepared seatbelt set plus two Roboflow YOLOv8 export zips (smoking, cap),
remaps every source class to the unified CabinSense taxonomy (smoking / seatbelt / cap)
via alias matching, and writes a single ``data/detect_custom`` dataset the consolidated
detector trains on. Run from the project root:

    python scripts/prepare_unified_detection.py \
        --smoking-zip "data/smoking.v1i.yolov8.zip" --cap-zip "data/cap dataset.v3i.yolov8.zip"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CUSTOM_CLASS_ALIASES,
    CUSTOM_DETECTOR_CLASSES,
    DATA_DIR,
)
from src.data.detection_prep import (
    build_source_id_map,
    ingest_split,
    load_yolo_names,
    write_unified_yaml,
)

DEST_ROOT = DATA_DIR / "detect_custom"
EXTRACT_ROOT = DATA_DIR / "_extract"
SEATBELT_DIR = DATA_DIR / "seatbelt_yolo"
SPLITS = ("train", "valid")
UNIFIED_INDEX = {name: i for i, name in enumerate(CUSTOM_DETECTOR_CLASSES)}


def _parse_args() -> argparse.Namespace:
    """Paths to the two Roboflow YOLOv8 export zips."""
    parser = argparse.ArgumentParser(description="Merge detection datasets")
    parser.add_argument("--smoking-zip", required=True)
    parser.add_argument("--cap-zip", required=True)
    return parser.parse_args()


def _extract_zip(zip_path: Path, tag: str) -> Path:
    """Extract a Roboflow zip to a clean per-source directory and return its root."""
    dest = EXTRACT_ROOT / tag
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    return dest


def _ensure_seatbelt() -> None:
    """Prepare the seatbelt dataset if it is not already on disk."""
    if (SEATBELT_DIR / "data.yaml").exists():
        return
    from scripts.prepare_seatbelt_data import main as prepare_seatbelt

    prepare_seatbelt()


def _ingest_source(root: Path, tag: str) -> None:
    """Remap and copy every split of one source dataset into the unified dataset."""
    id_map = build_source_id_map(load_yolo_names(root / "data.yaml"), CUSTOM_CLASS_ALIASES, UNIFIED_INDEX)
    print(f"{tag}: source names {load_yolo_names(root / 'data.yaml')} -> id map {id_map}")
    for split in SPLITS:
        copied = ingest_split(root / split, id_map, DEST_ROOT / split, tag)
        if copied:
            print(f"  {split}: {copied} images")


def main() -> None:
    """Build the merged smoking/seatbelt/cap detection dataset."""
    args = _parse_args()
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    _ensure_seatbelt()
    _ingest_source(SEATBELT_DIR, "seatbelt")
    _ingest_source(_extract_zip(Path(args.smoking_zip), "smoking"), "smoking")
    _ingest_source(_extract_zip(Path(args.cap_zip), "cap"), "cap")
    write_unified_yaml(DEST_ROOT, CUSTOM_DETECTOR_CLASSES)
    print(f"wrote {DEST_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
