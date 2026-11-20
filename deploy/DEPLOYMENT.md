# CabinSense - Deployment & SoC Portability Guide

CabinSense is implemented in Python, but engineered so a future C/C++ port to an
automotive SoC is mechanical rather than a rewrite. This document is that contract:
the artifacts, the runtime dependencies, the Python -> C/C++ mapping, and the
optimization options. No C/C++ is shipped in this repository.

## Artifacts

- `deploy/models/*.onnx` and `*.int8.onnx` - all learned components (drowsiness
  classifier, multi-task attribute model) plus the ONNX detector (YuNet) and
  landmark model (PFLD).
- `deploy/model_manifest.json` - the single source of truth: per-model input name,
  shape (NCHW), layout, color order, normalization (mean/std), output names and
  activation, plus all decision thresholds and temporal-window parameters.

## Runtime dependencies for a port

- **ONNX Runtime (C++ API)** - runs every learned model; one dependency for all.
- **OpenCV (C++)** - image IO, resize/color conversion, YuNet detector, `solvePnP`.
- A JSON parser - to read the manifest.

No Python, no PyTorch, no MediaPipe on the target.

## Execution providers (same graph, different silicon)

ONNX Runtime abstracts the accelerator behind an execution provider. The exported
graphs run unchanged on:

- `CPUExecutionProvider` (baseline)
- `CUDAExecutionProvider` (dev GPU)
- `TensorRTExecutionProvider` (NVIDIA automotive SoCs)
- vendor NPU providers (e.g. QNN, OpenVINO)

Selecting a provider is a session-option change, not a code change.

## Optimization ladder

1. **fp32** ONNX - reference.
2. **fp16** - halves size, ~free on GPU/NPU.
3. **INT8 dynamic** (`*.int8.onnx`) - calibration-free and 3.6x smaller, but 6-11x
   slower than fp32 on CPU for these backbones (`outputs/benchmark.json`). Use it only
   where flash or RAM is the binding constraint; fp32 is the default.
4. **INT8 static (QDQ)** - better conv speedups; add a calibration reader over a few
   hundred representative frames. Documented follow-on.
5. **TensorRT engine** - build from the ONNX for maximum throughput on NVIDIA SoCs.

## Python -> C/C++ mapping

The runtime path is deliberately dependency-light so each piece has a direct C++
equivalent:

| Python (`src/`) | Responsibility | C/C++ equivalent |
|---|---|---|
| `runtime/onnx_session.py` | model inference | `Ort::Session::Run` |
| `core/preprocess.py` | resize/normalize/CHW | `cv::resize`, `cv::cvtColor`, tensor fill |
| `runtime/detect.py` | face detection | OpenCV `FaceDetectorYN` (C++) |
| `runtime/landmarks.py` | 98-pt landmarks | ORT run + coordinate scaling |
| `core/features.py` | EAR/MAR/PERCLOS | plain float math |
| `core/pose.py` | head pose | `cv::solvePnP`, `cv::RQDecomp3x3` |
| `core/gaze.py` | gaze zone | `if/else` on angles |
| `core/temporal.py` | smoothing/hysteresis | ring buffer + counters |
| `core/fusion.py` | state decision | boolean logic |
| `core/manifest.py` | contract | JSON read + struct fill |

Because `scripts/verify_parity.py` guarantees the Python+ONNX path matches the
PyTorch models within tolerance, the ONNX artifacts are the frozen reference a C++
port reproduces.

## Suggested port steps

1. Load `model_manifest.json`; create one `Ort::Session` per model with the chosen EP.
2. Port `core/preprocess.py` and the EAR/MAR/pose/gaze/fusion/temporal functions
   (all small, side-effect-free).
3. Reproduce the per-frame -> temporal-fusion flow of `runtime/pipeline.py`.
4. Validate against the Python outputs on a shared clip (the parity contract).
