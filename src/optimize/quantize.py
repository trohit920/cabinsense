"""INT8 quantization of ONNX models via ONNX Runtime.

Dynamic INT8 quantization is the primary, calibration-free path (weights to int8),
which shrinks the model and speeds CPU/edge inference while keeping the graph and the
core logic unchanged. Static (QDQ) quantization and TensorRT are documented as
optional follow-ons in DEPLOYMENT.md.
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_int8(source_path: Path, output_path: Path) -> Path:
    """Dynamically quantize an ONNX model's weights to INT8; return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(source_path), str(output_path), weight_type=QuantType.QInt8)
    return output_path
