# CabinSense - Design & Methodology Report

A local, multi-task in-cabin driver & occupant monitoring system (DMS/OMS),
implemented in Python and packaged for SoC portability. This report covers the
approach, design decisions, measured results, and limitations. All numbers below
come from `outputs/eval.json` and `outputs/benchmark.json`, reproducible with the
commands in Section 7.

## 1. Problem

In-cabin monitoring must do more than classify a single frame: it must fuse several
weak signals into a stable decision, stay reliable under vehicle conditions
(lighting, occlusion, pose), and run within an automotive SoC's compute budget. The
bar that matters is "AI that actually operates in a vehicle" - so this project is
built as a deployable module, not a notebook.

## 2. System

Eleven outputs from one camera stream: drowsiness, distraction, gaze zone, yawning,
phone use, drinking, smoking, seatbelt, glasses, cap, and occupant count. The pipeline
is a multi-model fusion:

- ONNX **YuNet** face detection (also the occupant-count signal).
- ONNX **PFLD** 98-point landmarks -> EAR and MAR.
- **6DRepNet** head pose, with the 5-point `solvePnP` estimate as the fallback when
  the estimator is not loaded, and **L2CS** eye gaze.
- A learned **drowsiness** classifier (MobileNetV3) and a shared-backbone
  **multi-task attribute** model, trained with masked BCE for partial labels. Only
  glasses is still served from this model at inference.
- Two **localized object detectors**: COCO-pretrained YOLOv8n for phone and drinking,
  and one consolidated YOLOv8n fine-tuned on merged open detection sets for smoking,
  seatbelt, and cap.
- A **driver ROI with two-rate scheduling**: most frames process only the driver
  region (face -> torso -> lap); a full-frame sweep every few seconds recounts
  occupants and re-localizes the driver.
- Pure **fusion + temporal** logic (PERCLOS window, distraction window, hysteresis
  flags, majority-vote gaze/occupancy, confidence-gated abstention) producing a
  stable `CabinState`.

## 3. Key design decisions

- **ONNX-everywhere + dependency-light core.** Every learned stage is ONNX; all
  decision logic lives in `src/core/` using only ops with direct C++ equivalents.
  This is what makes the SoC port mechanical (see `deploy/DEPLOYMENT.md`).
- **Geometry + learning fusion.** EAR/PERCLOS and head-pose give robust, data-free
  baselines; the learned models refine them. Eye closure, for example, fires on low
  EAR *or* the learned signal, which is the "combine multiple models" requirement.
- **Masked multi-task learning.** Different public datasets label different tasks;
  masked BCE lets each sample supervise only its available tasks - directly the
  "sparse/imperfect GT" challenge, handled honestly rather than by faking labels.
- **No MediaPipe / no albumentations.** Both pull dependencies that conflict or fail
  to build on this Windows/ONNX toolchain; ONNX landmarks and torchvision-v2
  augmentation give a single clean, portable path.
- **Temporal stability as a first-class output.** Flicker is measured, and hysteresis
  prevents flag chatter - stability is one of the three metric axes, alongside
  accuracy and latency.
- **Detection over classification for object-like classes.** A whole-image classifier
  has no concept of absence: on a clip containing no phone, the phone head still
  reported p90 ~0.99. Object-like behaviours (phone, drinking, smoking, seatbelt, cap)
  moved to localized detectors, which answer "where" and can therefore answer "not
  present". Genuine face attributes stay classifiers - glasses is not a separable
  object, so a face-crop head is the right tool. Yawn stays MAR geometry.
- **One consolidated detector, not one per class.** The classes COCO lacks (smoking,
  seatbelt, cap) share a single fine-tuned YOLOv8n rather than three models, so the
  on-device inference budget stays flat as classes are added. Heterogeneous open
  datasets are merged through a case-insensitive class-alias map, so an export
  labelling its class `Cigarette` or `Vape` folds into `smoking` without hand-editing.
- **Open-vocabulary detection was tried and rejected.** YOLO-World was the obvious
  zero-shot route for the data-less classes, but it failed on grayscale cabin frames,
  so supervised fine-tuning is the path taken rather than the fallback.
- **ROI as an efficiency decision, not a cosmetic one.** Running detectors on the
  driver region rather than the full frame roughly halves the pixel budget and raises
  effective resolution on small objects, which is where phone and cigarette detection
  actually fails. The periodic full sweep is what keeps occupant count correct.

## 4. Datasets (public, no-auth; confirmed in Phase 1)

- Drowsiness: `akahana/Driver-Drowsiness-Dataset` (33k/8k).
- Glasses + cap: `huggan/CelebA-faces-with-attributes` (Eyeglasses, Wearing_Hat).
- Eye open/closed (alt): `MichalMlodawski/closed-open-eyes`.
- Distraction: head-pose geometry (no labeled open set found).
- Phone/drinking: COCO via a pretrained YOLOv8n (`cell phone`, `bottle`, `cup`) - no
  training data needed, because COCO already covers these classes.
- Smoking/seatbelt/cap detection: merged open Roboflow YOLOv8 exports, consolidated
  into one three-class set by `scripts/prepare_unified_detection.py`.
- Yawn: MAR geometry, no dataset required.
- No Kaggle credentials on this machine, so Kaggle-gated sets are excluded.

## 5. Evaluation methodology (accuracy / latency / stability)

- **Accuracy:** drowsiness accuracy/macro-F1 on a held-out test split; masked
  per-task accuracy/F1 for attributes (`scripts/run_eval.py`).
- **Robustness / failure analysis:** the same test set is re-scored under degraded
  (augmented lighting/occlusion/pose) conditions to quantify the drop, plus a saved
  confusion matrix for failure inspection.
- **Latency / cost:** `scripts/benchmark.py` reports median/p95 latency, FPS, and
  model size per ONNX model, fp32 vs INT8, across CPU/CUDA - the on-device budget.
- **Parity:** `scripts/verify_parity.py` asserts PyTorch and ONNX agree within 1e-3.
- All runs are seeded (`SEED = 42`) and tracked in MLflow.

## 6. Measured results

### 6.1 Accuracy (`outputs/eval.json`)

| Model | Metric | Clean | Degraded | Support |
|---|---|---|---|---|
| Drowsiness | accuracy | 0.9999 | 0.9988 | 8,359 |
| Drowsiness | macro-F1 | 0.9999 | 0.9988 | 8,359 |
| Multi-task: glasses | accuracy / F1 | 0.9889 / 0.9137 | - | 1,535 |
| Multi-task: cap | accuracy / F1 | 0.9837 / 0.7899 | - | 1,535 |

The drowsiness model makes **one error in 8,359 test images** (confusion matrix in
`outputs/figures/`), and loses only 0.011 percentage points under the full
augmentation stack. Both facts point the same way: **this dataset is saturated, and
the number should not be read as a generalization result.**
`akahana/Driver-Drowsiness-Dataset` is assembled from video frames, so visually
near-duplicate frames almost certainly appear on both sides of its official
train/test split. A model can score this well by recognizing subjects and scenes
rather than eyelid state. The correct next step is a subject-disjoint evaluation -
UTA-RLDD held out by participant id - which is what would actually measure whether
the eye-state signal transfers. Reporting 0.9999 without that caveat would be
misleading, so the caveat travels with the number everywhere it appears.

The attribute heads are the more honest signal. `glasses` reaches F1 0.914 on a
reasonably balanced class. `cap` reaches accuracy 0.984 but F1 0.790, the classic
imbalance gap: `Wearing_Hat` is roughly a 5% positive class in CelebA, so accuracy
is inflated by the negative majority and F1 is the metric that matters. The masked
loss means `seatbelt`, `phone`, `smoking`, and `yawn` contribute no gradient to this
model and report `NaN` at `support: 0` rather than a fabricated score. Those classes
are now served by the object detectors instead, which are evaluated as detectors and
are not part of this classification table.

### 6.2 Latency, throughput, and size (`outputs/benchmark.json`)

ONNX Runtime, `CPUExecutionProvider`, 100 iterations after warmup, random inputs at
each graph's own input shape (dynamic dimensions filled with the face-crop size).

| Model | Median | p95 | FPS | Size |
|---|---|---|---|---|
| yunet.onnx (detection) | 3.82 ms | 11.58 ms | 262 | 0.23 MB |
| drowsiness.onnx (fp32) | 2.07 ms | 4.13 ms | 483 | 6.09 MB |
| drowsiness.int8.onnx | 13.22 ms | 20.78 ms | 76 | 1.70 MB |
| multitask.onnx (fp32) | 1.47 ms | 2.20 ms | 682 | 6.11 MB |
| multitask.int8.onnx | 15.71 ms | 23.98 ms | 64 | 1.71 MB |

The fp32 chain (detect -> drowsiness -> attributes) costs ~7.4 ms per frame, about
135 FPS, on CPU with no accelerator. A 30 FPS camera leaves ~26 ms of headroom per
frame for landmarks, geometry, and fusion.

### 6.3 The INT8 result: 3.6x smaller, 6-11x slower

The quantization step produced a regression, and it is the most transferable finding
in the project.

Dynamic INT8 quantization stores weights as int8 and, **on every call, computes
activation scales and wraps each quantized operator in quantize and dequantize
steps.** That is a good trade when
the convolution arithmetic being accelerated dominates the runtime budget. It is a
bad trade here: MobileNetV3-Small at 128x128 already runs a 1.5-2 ms forward pass,
so the per-operator quantize/dequantize work is likely never amortized and becomes
the dominant cost (probable mechanism; no operator profile is saved). Measured: **6.4x slower** on the drowsiness model, **10.7x slower** on
the multi-task model, in exchange for a **3.6x** size reduction.

This changes the deployment recommendation rather than invalidating the work:

| Constraint | Right variant | Why |
|---|---|---|
| Compute-bound | fp32 | Strictly faster on this backbone; already meets 30 FPS with margin |
| Flash/RAM-bound | INT8 dynamic | About 8.8 MB saved across both models (3.6x smaller each), if ~15 ms/model is affordable |
| Both | INT8 static (QDQ) | Folds quantization into the graph, removing the per-call dequantize - needs a few hundred calibration frames |
| GPU/NPU target | fp16 | Native accelerator path on automotive silicon; the likely real win |

The generalizable lesson: **on already-small edge backbones, quantization is a size
optimization, and its effect on speed must be measured rather than assumed.** The
common intuition - "INT8 is faster" - is inherited from large models where compute
dominates, and it inverts at this scale. This is exactly why the project ships a
benchmark script rather than a claim.

## 7. Reproducing results

```bash
python scripts/download_models.py        # YuNet (+ place PFLD)
python scripts/train_drowsiness.py       # -> checkpoint + MLflow
python scripts/train_multitask.py        # -> checkpoint + MLflow
python scripts/run_eval.py               # -> outputs/eval.json + confusion figure
python scripts/export_and_quantize.py    # -> deploy/models/*.onnx (+ INT8)
python scripts/verify_parity.py          # PyTorch <-> ONNX parity
python scripts/benchmark.py              # -> outputs/benchmark.json
python scripts/build_manifest.py         # -> deploy/model_manifest.json
```

## 8. Limitations

These are ordered by how quickly a reviewer running the demo will hit them.

- **Driver selection is a heuristic and it is wrong on the sample clip.**
  `infer_drive_side` votes on the largest face during the auto-init sweeps, and the
  sample footage is right-hand-drive with the front-left passenger nearest the camera.
  The system locks LHD and tracks the passenger. Face area is a proxy for camera
  distance, not for seat position. A production system reads drive side from vehicle
  configuration; `--drive-side rhd` overrides it here. A better heuristic would use
  steering-wheel or A-pillar geometry rather than face size.
- **The committed demo clip predates a fusion fix.** It shows `DROWSY` at
  `p 0.53, EAR 0.30, PERCLOS 0.00`. Two defects produced that, both now fixed and
  covered by tests. First, `_fuse` ran the learned branch as a bare per-frame
  threshold (`is_drowsy(perclos) or eye_closed_prob >= 0.5`) while only the geometry
  branch was windowed; both now accumulate over the same PERCLOS horizon. Second, and
  more consequential, `RatioWindow.ratio()` divided by the number of samples seen
  rather than the window capacity, so the first True in a stream returned 1.0 - every
  ratio-driven flag, PERCLOS and distraction included, was trivially satisfiable on
  frame one. The ratio now counts unfilled slots as False and ramps in over the
  window, which errs toward "not yet triggered". Regenerate the clip with
  `scripts/demo_dms_video.py` to see the corrected behaviour.
- **RGB-to-grayscale domain gap on every detector.** The demo footage is grayscale;
  both occupants wear seatbelts and the seatbelt class reports 0 throughout, and the
  driver's glasses go unreported. The consolidated detector was fine-tuned on RGB
  Roboflow exports. Fine-tuning on IR/grayscale driver-facing footage is required
  before any of these classes carries a claim.
- **The drowsiness benchmark is saturated.** See Section 6.1: the dataset's
  train/test split does not appear to be subject-disjoint, so 0.9999 measures
  memorization capacity as much as eye-state discrimination. A subject-held-out
  re-evaluation is the single highest-value next experiment in this project.
- **Benchmark coverage is partial.** `outputs/benchmark.json` measures the ONNX
  models only. 6DRepNet, L2CS, and both YOLO detectors run in PyTorch and are
  excluded, so the ~7.4 ms chain figure is the face/attribute path, not the full
  overlay pipeline. Exporting those to ONNX and re-benchmarking is the follow-on that
  would make the latency story complete.
- **Geometry-based distraction/gaze/yawn** are baselines; learned refinements need
  labeled gaze/yawn sets (interfaces already in place).
- **Landmark indices** assume the WFLW-98 convention and must be re-validated against
  the specific PFLD export (flagged in `src/core/landmark_indices.py`).
- **Static INT8 (QDQ) and TensorRT** are documented follow-ons; dynamic INT8 is the
  shipped quantization, and Section 6.3 shows why static QDQ is the change that
  would actually make INT8 competitive on latency here.
