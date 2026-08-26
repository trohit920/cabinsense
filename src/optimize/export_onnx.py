"""Export trained PyTorch models to ONNX (fixed opset, static shapes).

Static input shapes and a pinned opset keep the graphs portable to a SoC's ONNX
Runtime. Dynamic batch is opt-in but off by default for deployment predictability.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import ONNX_DYNAMIC_BATCH, ONNX_OPSET


def export_model(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    input_name: str = "input",
    output_name: str = "output",
    opset: int = ONNX_OPSET,
    dynamic_batch: bool = ONNX_DYNAMIC_BATCH,
) -> Path:
    """Export a model to ONNX and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {input_name: {0: "batch"}, output_name: {0: "batch"}} if dynamic_batch else None
    )
    torch.onnx.export(
        model.eval(),
        dummy_input,
        str(output_path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    return output_path
