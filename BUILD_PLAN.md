# CabinSense - In-Cabin Driver & Occupant Monitoring (Build Plan / Design Doc)

The design document written before the code. It fixes the requirements, the task
taxonomy, the module boundaries, and the phase plan, and is preserved here
unedited-in-substance so the reasoning behind the build is inspectable alongside
the result.

**CabinSense** is a multi-task, real-time in-cabin monitoring system (DMS/OMS), built
**entirely in Python**, but packaged as a **deployment-ready, port-ready module**: all
learned components export to ONNX, the runtime logic is dependency-light and portable,
and a formal deployment contract plus a portability guide make a later C/C++ port to an
automotive SoC straightforward *if it is ever needed*. No C/C++ is written in this
project - only the design and artifacts that would make such a port low-risk.

Constraints: fully local, open-source models and datasets, no paid APIs, developed
on a ~6 GB NVIDIA GPU. House style matches the companion project
(`agentic-rag-guardrails`): `src/` library + `config.py` + `scripts/` + `tests/` +
`outputs/`, SEED = 42, ruff + pytest gates, report + notebook.

---

## 1. Goal

Infer, per camera frame and stably over time, a rich in-cabin state:

- **Driver state (temporal):** attentive / drowsy / distracted
- **Gaze & alertness:** gaze zone (where the driver is looking), yawning
- **Behaviors:** phone use, smoking
- **Attributes:** wearing glasses, wearing cap/hat, seatbelt fastened
- **Occupancy (OMS):** people / occupant count in the cabin

...and deliver it not as a notebook demo but as a **ready-to-deploy module**: exported
INT8 ONNX models, a model manifest, a portable inference core, a C++ reference, and a
documented optimization/verification process - the "AI that actually operates in a
vehicle" that is the actual bar for in-cabin sensing work.

---

## 2. Task taxonomy (what is inferred, and how)

| Task | Type | Signal / model | Region |
|---|---|---|---|
| Drowsiness | temporal | eye-state CNN + EAR + PERCLOS | face / eyes |
| Distraction | temporal | head-pose (yaw/pitch) + gaze off-road duration | face |
| Gaze zone | per-frame + temporal | head-pose + eye gaze -> zone (geometry baseline, optional learned) | face / eyes |
| Yawning | temporal | MAR + yawn classifier over a time window | face / mouth |
| Phone use | behavior | multi-task classifier head (+ optional detection) | driver crop |
| Smoking | behavior | multi-task classifier head | face / driver crop |
| Seatbelt fastened | attribute | multi-task classifier head | upper-body crop |
| Glasses | attribute | multi-task classifier head | face |
| Cap / hat | attribute | multi-task classifier head | head / upper face |
| People count (OMS) | per-frame | face/person detector count + temporal stabilization | full cabin |

This is a genuine **multi-model, multi-task, multi-label** system, which directly
exercises the "final decision quality when combining multiple models" challenge.

---

## 3. Requirement -> feature traceability

The capability requirements for a production in-cabin sensing system, each mapped
to the module that delivers it. Building this table first is what kept the scope
honest: every row is either implemented or explicitly marked pending data.

| Requirement | Feature | Module |
|---|---|---|
| Design & train DL models for vehicle conditions | Trained eye-state + multi-task attribute models with lighting/occlusion/pose augmentation | `train/`, `data/transforms.py` |
| Detection, Landmark, Pose, Classification, Segmentation | Face detection (ONNX) + landmarks (ONNX/MediaPipe) + head-pose + multi-task classification; segmentation = documented extension | `runtime/`, `models/` |
| SOTA analysis/application/improvement | EAR/PERCLOS paper baselines improved by learned heads; documented deltas | `core/features.py`, `report.md` |
| Failure-case analysis + root cause | Per-task, per-condition confusion matrices + hard-example galleries (data/model/env) | `eval/`, `scripts/run_failure_analysis.py` |
| accuracy / latency / stability metrics | Per-task accuracy/F1; latency ms + FPS; temporal-flicker stability | `eval/metrics.py`, `scripts/benchmark.py`, `core/temporal.py` |
| Result-driven improvement | Backbone / augmentation / fusion / temporal ablations with logged deltas | `scripts/run_ablation.py` |
| On-board SoC lightweighting | ONNX export + INT8 (+ fp16); optional TensorRT/other backends via ORT EPs | `optimize/`, `deploy/` |
| Inference-pipeline & system verification | End-to-end frame->state pipeline + python<->ONNX parity test + video verification | `runtime/pipeline.py`, `scripts/verify_parity.py` |
| Reproducible experiment management | MLflow local tracking; SEED=42; fixed splits | `train/`, MLflow |
| Repeatable eval/verification | Single-command seeded eval harness | `scripts/run_eval.py` |
| Inference under limited compute | Small backbone + INT8; memory/latency budget reported | `models/`, `scripts/benchmark.py` |
| Latency/accuracy balance | Accuracy-vs-latency Pareto across fp32/fp16/INT8 and backbones | `scripts/run_ablation.py` |
| Reliable eval under imperfect/sparse GT | Partial-label multi-task training (masked loss) + confidence-gated abstention | `train/`, `core/temporal.py` |
| Multi-model decision quality | Fusion of geometry (EAR/pose) + learned heads into one state | `core/fusion.py` |
| CNN + Transformer understanding | MobileNetV3 default + MobileViT variant in the ablation | `models/` |
| **Deploy-ready / SoC-portable** | ONNX-only inference, dependency-light portable core, model manifest, DEPLOYMENT.md portability guide | `deploy/`, `src/core/` |

---

## 4. Architecture / data flow

```
camera frame (webcam or video)
  -> face / driver-region detection      (ONNX detector: YuNet)
  -> facial landmarks                    (ONNX landmark model: PFLD 98-pt)
  -> geometry: EAR, MAR, head-pose       (pure core/, deterministic)
  -> crops: eyes / face / upper-body
  -> eye-state CNN (ONNX)                 -> drowsiness signal
  -> multi-task attribute CNN (ONNX)      -> glasses / cap / seatbelt / phone / smoking
  -> fusion (core/fusion.py)             -> per-frame state + per-task confidences
  -> temporal smoothing (core/temporal)  -> PERCLOS window, hysteresis, abstain-on-low-conf
  -> STABLE STATE: {drowsy, distracted, gaze_zone, yawning, phone, smoking,
                    seatbelt, glasses, cap, occupant_count} + confidences
```

The learned stages are ONNX; everything between them (`core/`) is pure, deterministic,
dependency-light logic that maps 1:1 to a later C++ port.

---

## 5. Deployment and SoC portability (Python-built, port-ready by design)

The project is written entirely in Python, but structured so the runtime path could
later be re-implemented in C/C++ on a SoC with no surprises. Nothing here is built in
C/C++; the goal is that a future port would be mechanical:

- **ONNX everywhere.** Every learned component (detector, landmarks, eye-state,
  multi-task attributes) is exported to ONNX with a fixed opset and static input
  shapes, plus INT8-quantized variants. Deployment uses **ONNX Runtime**, which has a
  first-class C++ API and pluggable execution providers (CPU, CUDA, TensorRT, and
  vendor NPUs) - so the same graph targets different SoCs by swapping the EP.
- **ONNX landmarks throughout (no MediaPipe).** Detection and landmarks are ONNX
  models (YuNet + PFLD) in both prototyping and deployment, so there is a single
  ORT-only path and no MediaPipe/protobuf dependency to reconcile.
- **Portable core.** `src/core/` (preprocess, EAR/MAR/PERCLOS, pose, fusion, temporal)
  uses only numpy / OpenCV ops that have direct C++ equivalents - no pandas, sklearn,
  or Python-only constructs in the runtime path. This is the reference a future C++
  port would mirror line-for-line.
- **Deployment contract (`deploy/model_manifest.json`).** For each model: file, input
  name/shape/layout (NCHW), dtype, color order, resize dims, mean/std or scale, output
  names, class maps, and all decision thresholds and temporal-window parameters. A C++
  engineer needs nothing beyond this file and the ONNX models to reproduce behavior.
- **Portability guide (`deploy/DEPLOYMENT.md`).** Documents exactly how a future
  C/C++ SoC port would consume the manifest and ONNX models: which ORT C++ API calls
  map to the Python `runtime/onnx_session.py`, how each `core/` function (preprocess,
  EAR/MAR/PERCLOS, pose, fusion, temporal) translates to C++, and the memory/latency
  budget. It is a specification for a port, not a port.
- **Parity guarantee.** `scripts/verify_parity.py` asserts the Python `core/` + ORT
  path matches the training-framework (PyTorch) outputs within tolerance, so the ONNX
  artifacts are the frozen source of truth any future port would reproduce.
- **Optimization ladder.** fp32 -> fp16 -> INT8 (ORT static quantization) as the
  primary path; TensorRT and other quantizers wired as optional ORT execution
  providers / backends "on need" without changing the graph or the core logic.

`deploy/DEPLOYMENT.md` documents the build, the contract, the EP options, and the
memory/latency budget - i.e. "all requirements: model, library, optimization done."

---

## 6. Tech stack (all local) + VRAM budget

| Concern | Choice | Why |
|---|---|---|
| Training | PyTorch (CUDA cu126) + torchvision (+ timm for backbones) | Standard, matches the companion project's venv |
| Backbones | MobileNetV3-Small (default); MobileViT (CNN-vs-Transformer ablation) | edge-appropriate, fit 6 GB |
| Face detection | ONNX YuNet (or Ultra-Light-Fast) | small, ORT/OpenCV, edge-friendly |
| Facial landmarks | ONNX PFLD (98-pt) | dense enough for EAR/MAR/pose; ONNX-only, no MediaPipe (avoids protobuf conflicts) |
| Head pose | OpenCV solvePnP | deterministic, C++-available |
| Augmentation | torchvision.transforms.v2 + OpenCV | lighting / occlusion / pose robustness (no albumentations - its stringzilla dep fails to build on this Windows toolchain) |
| Runtime / optimization | onnx, onnxruntime-gpu, ORT quantization (INT8) | deploy path with C++ API |
| Experiment tracking | MLflow (local file backend) | Free, offline, local file backend |
| Metrics / plots | scikit-learn, matplotlib, seaborn | training/eval only, not runtime |
| Video / IO | opencv-python | webcam + video demos |

VRAM: MobileNetV3-Small multi-task at 128-224px, batch <=64, fp16 - well within 6 GB.

---

## 7. Datasets (public, multi-dataset, partial-label; verify-at-build)

Different tasks come from different public datasets; the multi-task model trains with
**masked loss** so each sample contributes only to its available labels (this is also
the "sparse/imperfect GT" challenge, handled honestly).

| Task | Candidate public datasets |
|---|---|
| Drowsiness / eye-state | MRL Eye Dataset; UTA-RLDD; NTHU-DDD; "Driver Drowsiness (DDD)" |
| Yawning | YawDD (Yawning Detection Dataset); UTA-RLDD (yawn segments) |
| Gaze zone | DGW (Driver Gaze in the Wild); MPIIGaze; Columbia Gaze; LISA Gaze (geometry baseline needs no data) |
| Distraction / phone | State Farm Distracted Driver; AUC Distracted Driver (AUCD2); NHTSA SynDD2 |
| Glasses + cap/hat | CelebA (attributes `Eyeglasses`, `Wearing_Hat`) |
| Smoking | public "smoking / not-smoking" image datasets (Kaggle/HF mirrors) |
| Seatbelt | public seatbelt-detection datasets; SynDD2 activities where available |
| People count (OMS) | face detector (WIDER FACE-trained) count; multi-face clips for verification |
| Eval video | public driving/face clip or self-recorded webcam |

Phase 1 verifies availability and licensing (public-first, the same discipline as the companion project).
Each attribute head is enabled once its dataset is confirmed; tasks without a clean
open dataset are clearly marked "architecture-ready, pending data" rather than faked.
Distraction has an always-available fallback: derived from head-pose geometry even
without a labeled dataset.

**Confirmed in Phase 1 (HuggingFace, no auth required; no Kaggle credentials on this
machine, so Kaggle-gated sets are excluded):**
- Drowsiness: `akahana/Driver-Drowsiness-Dataset` (33,434 train / 8,359 test,
  ClassLabel[Drowsy, Non Drowsy], driver faces).
- Eye open/closed: `MichalMlodawski/closed-open-eyes` (126,560, per-eye state).
- Glasses + cap/hat: `huggan/CelebA-faces-with-attributes` (202,599, with `Eyeglasses`
  and `Wearing_Hat` binary attributes).
- Distraction: head-pose geometry (no labeled open set found;
  `gymprathap/Driver-Distracted-Dataset` has images but no labels).
- Phone / smoking / seatbelt / yawning: architecture-ready, pending labeled open data
  (yawning served by MAR geometry in the interim). `chbh7051/driver-drowsiness-dataset`
  is gated and excluded.

---

## 8. Module design (SRP, <500 lines/file, typed; core = C++-portable)

```
src/
  config.py            constants, paths, SEED, input sizes, thresholds, class maps
  core/                # deployment-portable, pure, minimal-dep (mirrors C++)
    preprocess.py      resize / normalize / color-order (deterministic)
    features.py        EAR / MAR / PERCLOS (pure, unit-tested)
    pose.py            head pose via solvePnP
    fusion.py          multi-signal -> per-frame state + confidences (pure)
    temporal.py        sliding window, hysteresis, abstain-on-low-confidence (pure)
    manifest.py        load/validate model_manifest.json
  models/
    multitask.py       shared backbone + per-task heads (masked multi-label)
    eye_state.py       eye-state classifier
  data/
    datasets.py        per-dataset loaders + unified partial-label sampler
    transforms.py      train/eval + lighting/occlusion/pose augmentation
  runtime/
    onnx_session.py    ORT inference wrapper (mirrors C++ ORT usage)
    detect.py          ONNX face/driver-region detector (YuNet)
    landmarks.py       ONNX landmark model (PFLD 98-pt)
    pipeline.py        frame -> stable state (uses core/ + runtime/)
  train/
    train_eye_state.py
    train_multitask.py
  eval/
    metrics.py         accuracy/F1/confusion, latency, stability (pure where possible)
    evaluate.py
  optimize/
    export_onnx.py     torch -> ONNX (fixed opset, static shapes)
    quantize.py        ORT INT8 static quantization
deploy/
  models/              *.onnx (+ *.int8.onnx)
  model_manifest.json  the deployment contract (shapes, norm, class maps, thresholds)
  DEPLOYMENT.md        portability guide: contract, ORT EP options, latency/memory budget,
                       and the Python->C/C++ mapping a future SoC port would follow
scripts/
  train_eye_state.py, train_multitask.py, run_eval.py, run_ablation.py,
  run_failure_analysis.py, export_and_quantize.py, benchmark.py,
  verify_parity.py, demo_video.py, demo_webcam.py
tests/                 pure-logic unit tests (features, temporal, fusion, metrics, manifest, parity)
outputs/               metrics, figures, checkpoints
```

---

## 9. Phased milestones (each ends with a runnable checkpoint)

- **Phase 0 - Scaffold + venv**: structure, config, requirements, CUDA torch venv,
  typed stubs. Checkpoint: imports + `pytest` collect pass.
- **Phase 1 - Data**: verify/download datasets, loaders, unified partial-label
  sampler, augmentation, sample grids. Checkpoint: dataset stats + augmented samples.
- **Phase 2 - Eye-state / drowsiness model**: train, MLflow-tracked. Checkpoint: val
  accuracy/F1 + checkpoint saved.
- **Phase 3 - Multi-task attribute model**: shared backbone + heads (glasses/cap/
  seatbelt/phone/smoking as data allows), masked loss. Checkpoint: per-task metrics.
- **Phase 4 - Pipeline + fusion + temporal**: detector+landmarks+pose+heads fused;
  temporal smoothing + abstention. Checkpoint: video -> stable multi-task timeline.
- **Phase 5 - Robustness + failure analysis**: lighting/occlusion/pose shifts,
  per-condition metrics, hard-example galleries, root-cause writeup. Checkpoint:
  robustness report + figures.
- **Phase 6 - On-device optimization**: ONNX export (all models), INT8 (+fp16),
  latency/FPS/size benchmark, accuracy-vs-latency Pareto, python<->ONNX parity.
  Checkpoint: benchmark table + Pareto + parity test green.
- **Phase 7 - Deployment packaging**: model_manifest.json + validation, DEPLOYMENT.md
  portability guide (Python->C/C++ mapping, ORT EP options, budgets), python<->ONNX
  parity. Checkpoint: manifest validates + parity test green on a sample frame.
- **Phase 8 - Write-up + demos**: report.md, analysis.ipynb, README, webcam + video
  demos (+ GIF), resume framing. Checkpoint: repo reads cleanly top to bottom.

Estimated ~3-4 focused weeks (multi-task training, multi-dataset, and the C++/deploy
layer make this the larger of the two POCs).

---

## 10. Testing

- Pure logic (EAR/MAR/PERCLOS, head-pose on synthetic points, fusion rules, temporal
  smoothing/hysteresis, metrics, manifest validation, split determinism) - seeded unit
  tests, no I/O.
- **Parity test**: Python `core/`+ORT vs training-framework outputs within tolerance
  (guards the deployment contract).
- Model-dependent paths via small smoke runs.
- Self-improving loop: write -> `pytest -q` + `ruff check` -> fix -> repeat.

---

## 11. Risks & mitigations

- **Dataset access / gating**: verify open mirrors in Phase 1; per-task fallback (e.g.
  distraction from head-pose geometry; CelebA for glasses/cap which is reliably
  available). Mark unfunded tasks "architecture-ready, pending data" - never faked.
- **Partial labels across datasets**: masked multi-task loss; report per-task support.
- **MediaPipe on Windows/Py3.11**: only a prototyping dep; deploy path is ONNX
  landmarks, so a failure doesn't block deployment.
- **TensorRT on Windows**: optional ORT execution provider "on need"; INT8 via ORT is
  the primary, always-available path.
- **Scope creep into C/C++**: explicitly out of scope. Deliverables are Python +
  ONNX artifacts + a portability guide; the parity test guarantees the ONNX path any
  future port would target. No C/C++ is written here.
- **6 GB VRAM**: small backbone, <=224px, batch<=64, fp16.
- **Class imbalance / weak GT**: class weighting + confidence-gated abstention.

---

## 12. Resume / GitHub framing

- Repo: `cabinsense` - "In-Cabin Driver & Occupant Monitoring, edge-deployable."
- One-line: "Multi-task DMS/OMS (drowsiness, distraction, phone, smoking, seatbelt,
  glasses, cap) - PyTorch, ONNX/INT8, multi-model fusion with temporal stability,
  packaged for SoC portability with a model manifest and deployment guide."
- Resume bullets: per-task accuracy/F1; latency/FPS fp32 vs INT8; model size; robustness
  deltas under lighting/occlusion; and "deploy-ready: ONNX + INT8 + manifest + portability guide."
- Pairs with the companion project (`agentic-rag-guardrails`): GenAI/LLM breadth +
  on-device CV depth.

---

## 13. Open questions resolved before Phase 0

1. Name **CabinSense** (repo `cabinsense`) - keep, or prefer another?
2. Confirm scope order: land drowsiness + distraction + glasses/cap first (data is
   reliable), then seatbelt/phone/smoking as datasets are confirmed - acceptable?
3. Anything to add to the state set beyond drowsiness, distraction, phone, smoking,
   seatbelt, glasses, cap (e.g. gaze zone, yawning as its own output, occupant count)?
