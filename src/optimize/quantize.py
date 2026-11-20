"""INT8 quantization of ONNX models via ONNX Runtime.

Dynamic INT8 quantization is the calibration-free path (weights to int8). It shrinks
the classifiers about 3.6x, but on CPU it measured 6-11x slower than fp32 for these
small MobileNetV3 backbones (outputs/benchmark.json), so fp32 is the runtime default
and INT8 is only for flash- or RAM-bound targets. Static (QDQ) quantization and TensorRT are documented as
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
