"""Deployment manifest: load and validate the model contract (pure).

The manifest (`deploy/model_manifest.json`) is the single source of truth a future
C/C++ SoC port consumes: per-model file, input name/shape/layout, color order,
normalization, and output names, plus global thresholds and temporal parameters.
This module loads and validates it - no model or framework dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_MODEL_KEYS: tuple[str, ...] = (
    "file",
    "input_name",
    "input_shape",
    "layout",
    "color",
    "mean",
    "std",
    "output_names",
)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the deployment manifest JSON from disk."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty means the manifest is valid)."""
    errors: list[str] = []
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        return ["manifest must contain a non-empty 'models' object"]
    for name, spec in models.items():
        if not isinstance(spec, dict):
            errors.append(f"model '{name}' spec must be an object")
            continue
        for key in REQUIRED_MODEL_KEYS:
            if key not in spec:
                errors.append(f"model '{name}' missing required key '{key}'")
        shape = spec.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 4:
            errors.append(f"model '{name}' input_shape must be a 4-element list (NCHW)")
    return errors
