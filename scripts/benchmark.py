"""Benchmark exported ONNX models: latency, FPS, and file size (Phase 6).

Times each model in deploy/models/ on random inputs (CPU and CUDA where available)
and records median/p95 latency, FPS, and model size. Writes outputs/benchmark.json,
supporting the accuracy-vs-latency and on-device-cost story.

Run from the project root:
    python scripts/benchmark.py --iters 100
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import DEPLOY_MODELS_DIR, FACE_INPUT_SIZE, OUTPUT_DIR
from src.eval.metrics import latency_stats
from src.runtime.onnx_session import OnnxModel


def _input_shape(model: OnnxModel) -> tuple[int, ...]:
    """Resolve a concrete input shape, defaulting dynamic dims to face-crop size."""
    raw = model.session.get_inputs()[0].shape
    dims = [d if isinstance(d, int) else s for d, s in zip(raw, (1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE))]
    return tuple(dims)


def _benchmark(model: OnnxModel, iters: int) -> dict[str, float]:
    """Warm up then time ``iters`` inferences on random input."""
    tensor = np.random.rand(*_input_shape(model)).astype(np.float32)
    for _ in range(5):
        model.run(tensor)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model.run(tensor)
        times.append((time.perf_counter() - start) * 1000.0)
    return latency_stats(times)


def main() -> None:
    """Benchmark every ONNX model in the deploy directory."""
    parser = argparse.ArgumentParser(description="Benchmark ONNX models")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for onnx_path in sorted(DEPLOY_MODELS_DIR.glob("*.onnx")):
        model = OnnxModel(onnx_path)
        stats = _benchmark(model, args.iters)
        stats["size_mb"] = round(onnx_path.stat().st_size / 1e6, 2)
        stats["providers"] = model.session.get_providers()
        results[onnx_path.name] = stats
        print(f"{onnx_path.name}: {stats}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
