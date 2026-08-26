"""ONNX Runtime inference wrapper.

A thin wrapper over onnxruntime.InferenceSession that mirrors how a C++ port would
use the ORT C++ API (create session, bind input by name, run, read outputs). Prefers
the CUDA execution provider and falls back to CPU. Keeping ORT usage behind this
class isolates the one runtime dependency the deployment target needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxModel:
    """Loads an ONNX model and runs single-input inference."""

    def __init__(self, model_path: Path, providers: list[str] | None = None) -> None:
        """Create an inference session, preferring CUDA then CPU."""
        available = ort.get_available_providers()
        default = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers or default)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Run inference on a single NCHW float32 input and return raw outputs."""
        return self.session.run(self.output_names, {self.input_name: tensor})
