"""Helpers to merge heterogeneous YOLO detection datasets into one taxonomy.

Each source dataset (a Roboflow YOLOv8 export or the prepared seatbelt set) labels its
own class ids against its own names. These functions resolve each source class to a
unified CabinSense class via case-insensitive aliases, remap label files to the unified
ids (dropping unmatched boxes so negatives become background), and copy image/label
pairs into a single YOLO layout. The pure functions carry the logic and are unit-tested;
the I/O functions are thin wrappers around them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resolve_alias(source_name: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    """Return the unified class whose alias list contains ``source_name`` (case-insensitive)."""
    needle = source_name.strip().lower()
    for unified, names in aliases.items():
        if needle in (n.lower() for n in names):
            return unified
    return None


def build_source_id_map(
    source_names: dict[int, str],
    aliases: dict[str, tuple[str, ...]],
    unified_index: dict[str, int],
) -> dict[int, int]:
    """Map source class ids to unified class ids for classes that resolve via aliases."""
    id_map: dict[int, int] = {}
    for src_id, src_name in source_names.items():
        unified = resolve_alias(src_name, aliases)
        if unified is not None and unified in unified_index:
            id_map[src_id] = unified_index[unified]
    return id_map


def remap_label_lines(lines: list[str], id_map: dict[int, int]) -> list[str]:
    """Rewrite YOLO label lines to unified class ids, dropping unmapped classes."""
    remapped: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        src_id = int(float(parts[0]))
        if src_id not in id_map:
            continue
        remapped.append(" ".join([str(id_map[src_id]), *parts[1:]]))
    return remapped


def load_yolo_names(data_yaml: Path) -> dict[int, str]:
    """Read a YOLO data.yaml ``names`` field (list or dict form) as an id->name dict."""
    doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = doc["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _label_for_image(image_path: Path) -> Path:
    """Locate the YOLO label .txt matching an image (images/ -> labels/)."""
    labels_dir = image_path.parent.parent / "labels"
    return labels_dir / (image_path.stem + ".txt")


def ingest_split(
    source_split_dir: Path,
    id_map: dict[int, int],
    dest_split_dir: Path,
    tag: str,
) -> int:
    """Copy one split's images and remapped labels into the unified dataset.

    Images with no surviving boxes are still copied as negatives (reduce false
    positives). Filenames are prefixed with ``tag`` to avoid cross-source collisions.
    Returns the number of images copied.
    """
    images_src = source_split_dir / "images"
    if not images_src.is_dir():
        return 0
    (dest_split_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_split_dir / "labels").mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in images_src.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = _label_for_image(image_path)
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        remapped = remap_label_lines(lines, id_map)
        stem = f"{tag}_{image_path.stem}"
        shutil.copyfile(image_path, dest_split_dir / "images" / (stem + image_path.suffix))
        (dest_split_dir / "labels" / (stem + ".txt")).write_text(
            "\n".join(remapped), encoding="utf-8"
        )
        count += 1
    return count


def write_unified_yaml(dest_root: Path, class_names: tuple[str, ...]) -> None:
    """Write the merged dataset's data.yaml with absolute path and unified names."""
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    dest_root.joinpath("data.yaml").write_text(
        f"path: {dest_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
