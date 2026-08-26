"""Download the pretrained ONNX detector/landmark models (Phase 1/4).

Fetches the YuNet face detector from the OpenCV Zoo (served via git-LFS media). The
PFLD 98-point landmark model is not hosted at a stable no-auth URL, so this script
prints clear placement instructions if it is absent; the pipeline requires it for
EAR/MAR geometry.

Run from the project root:
    python scripts/download_models.py
"""

from __future__ import annotations

import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DEPLOY_MODELS_DIR

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _download(url: str, destination: pathlib.Path) -> None:
    """Download a file to ``destination`` unless it already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"exists: {destination}")
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"saved: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    """Download YuNet and report PFLD placement instructions if missing."""
    _download(YUNET_URL, DEPLOY_MODELS_DIR / "yunet.onnx")
    pfld = DEPLOY_MODELS_DIR / "pfld.onnx"
    if not pfld.exists():
        print(
            "\nPFLD landmark model missing. Place a 98-point PFLD ONNX at:\n"
            f"  {pfld}\n"
            "Suitable exports are available from public PFLD/WFLW repositories; "
            "validate the eye/mouth landmark indices in src/core/landmark_indices.py."
        )


if __name__ == "__main__":
    main()
