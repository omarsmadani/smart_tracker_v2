# Edge Tracker — Occlusion-Robust Object Tracking on Coral Edge TPU

## Project Overview

Lightweight single-object tracker for the Coral Dev Board Mini using detection-association paradigm with occlusion handling. No SAM2. No GPU. Runs entirely on Edge TPU (INT8 TFLite) + CPU.

The algorithmic layer (state machine, smart memory, appearance recovery) is adapted from the SAMURAI occlusion fork. The inference backbone is replaced: SAM2 → SSD-MobileNetV2 (detection) + MobileNetV3-Small (feature extraction).

**Developer:** Solo
**Dev hardware:** Laptop with TFLite CPU runtime (development) → Coral Dev Board Mini (deployment)
**Target FPS:** ≥15 FPS on Coral
**Object scope:** COCO 80 classes (user selects target from detected objects)
**Input:** Recorded video files and live camera (USB/CSI)

---

## Design Principles

1. Don't overengineer: Simple beats complex
2. No fallbacks: One correct path, no alternatives
3. One way: One way to do things, not many
4. Clarity over compatibility: Clear code beats backward compatibility
5. Throw errors: Fail fast when preconditions aren't met
6. No backups: Trust the primary mechanism
7. Separation of concerns: Each function should have a single responsibility
8. Surgical changes only: Make minimal, focused fixes
9. Evidence-based debugging: Add minimal, targeted logging
10. Fix root causes: Address the underlying issue, not just symptoms
11. Simple > Complex
12. Collaborative process: Work with user to identify most efficient solution

---

## Architecture

### How It Works (per-frame loop)

```
1. Camera frame arrives
       │
       ▼
2. SSD-MobileNetV2 (Edge TPU) ──► all bounding boxes + confidence scores
       │
       ▼
3. Kalman predicts target position ──► IoU match best detection to prediction
       │
       ▼
4. MobileNetV3-Small (Edge TPU) ──► feature vector from matched bbox crop
       │
       ▼
5. State machine reads detection confidence ──► TRACKING / UNCERTAIN / LOST / RECOVERED
       │
       ├── TRACKING ──────► STM writes feature, LTM updates if best-ever
       ├── UNCERTAIN ─────► STM selective write (conf > 0.5), LTM frozen
       ├── LOST ──────────► Appearance recovery: compare fingerprint vs ALL detections
       └── RECOVERED ─────► Reset Kalman, keep STM + add recovery frame, keep LTM
       │
       ▼
6. Draw bbox (color by state) + log
```

### Key Difference from SAMURAI Fork

In the SAMURAI version, SAM2 propagates a segmentation mask forward using memory attention — one heavy model does everything. In this version, a fast detector runs every frame and we match detections to the target using motion (Kalman IoU) + appearance (MobileNetV3 features). Detection IS tracking. The occlusion intelligence layer on top is identical.

---

## Hardware: Coral Dev Board Mini

| Spec | Value |
|---|---|
| CPU | MediaTek 8167S, quad-core Cortex-A35 @ 1.5GHz |
| RAM | 2 GB LPDDR3 (shared CPU + Edge TPU) |
| ML accelerator | Google Edge TPU, 4 TOPS INT8 |
| Storage | 8 GB eMMC |
| Camera | MIPI-CSI2 (5MP Coral camera module) |
| Models supported | TFLite INT8 compiled with Edge TPU compiler |
| OS | Mendel Linux (Debian-based) |
| Power | USB-C 5V/2A |

### Memory Budget (~500MB of 2GB)

| Component | Memory |
|---|---|
| SSD-MobileNetV2 (INT8 TFLite) | ~15MB |
| MobileNetV3-Small (INT8 TFLite) | ~8MB |
| TFLite runtime | ~30MB |
| OpenCV + frame buffers | ~100MB |
| Python + our logic | ~50MB |
| OS (Mendel Linux) | ~300MB |

### Expected FPS

| Component | Edge TPU FPS | Notes |
|---|---|---|
| SSD-MobileNetV2 detection | 60-100 | At 300x300 input |
| MobileNetV3-Small features | 80+ | On bbox crop |
| Combined pipeline | 35-40 | Sequential on same TPU |
| After CPU overhead | 20-30 | Kalman + state machine + matching |

---

## Models

### SSD-MobileNetV2 (Detector)
- **Purpose:** Detect all objects in frame every frame
- **Input:** 300x300 RGB image
- **Output:** Bounding boxes + class labels + confidence scores
- **Model:** Pre-compiled Edge TPU TFLite from Coral model zoo
- **Download:** `https://coral.ai/models/object-detection/`
- **Classes:** COCO 80 classes
- **Role in pipeline:** Provides candidate bounding boxes. During TRACKING, the best IoU-matched detection is the target. During LOST, all detections are candidates for appearance recovery.

### MobileNetV3-Small (Feature Extractor)
- **Purpose:** Extract appearance feature vector from bbox crop
- **Input:** 224x224 RGB crop of target bounding box
- **Output:** 1024-dimensional feature vector (from last pooling layer, before classification head)
- **Model:** MobileNetV3-Small trained on ImageNet, quantized to INT8, compiled for Edge TPU
- **Role in pipeline:** Produces feature vectors stored in STM/LTM. Used for cosine similarity matching during appearance recovery. Replaces SAM2's image encoder features.

### No Training Required
Both models use pretrained weights. No fine-tuning. Download, compile for Edge TPU, deploy.

---

## Repository Structure

```
edge-tracker/
├── models/                         # TFLite model files
│   ├── ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
│   └── mobilenet_v3_small_quant_edgetpu.tflite
├── tracker/                        # Core tracking logic
│   ├── __init__.py
│   ├── detector.py                 # SSD-MobileNetV2 wrapper
│   ├── feature_extractor.py        # MobileNetV3-Small wrapper
│   ├── kalman_filter.py            # Kalman filter (predict + correct)
│   ├── data_association.py         # IoU matching between detections and prediction
│   ├── state_machine.py            # Occlusion state machine (copied from SAMURAI fork, adapted)
│   ├── smart_memory.py             # STM + LTM (copied from SAMURAI fork, adapted)
│   └── appearance_recovery.py      # Cosine matching recovery (copied from SAMURAI fork, adapted)
├── pipeline/                       # Main tracking pipeline
│   ├── __init__.py
│   └── tracker_pipeline.py         # Orchestrates all components per frame
├── visualization/                  # Display and logging
│   ├── __init__.py
│   └── display.py                  # Bbox overlay with state colors + trajectory logging
├── configs/
│   └── tracker_params.yaml         # All thresholds and parameters
├── scripts/
│   ├── run_video.py                # Run on recorded video file
│   ├── run_camera.py               # Run on live camera
│   └── select_target.py            # First-frame object selection UI
├── tests/                          # Unit tests
│   ├── test_state_machine.py
│   ├── test_smart_memory.py
│   ├── test_data_association.py
│   └── test_appearance_recovery.py
├── data/                           # Test videos
│   └── sample_videos/
├── claude.md                       # This file
├── requirements.txt
└── README.md
```

---

## What Each File Does

### tracker/detector.py
Wraps TFLite interpreter for SSD-MobileNetV2. One method: `detect(frame) -> list[Detection]` where Detection = namedtuple(bbox, class_id, confidence). Handles preprocessing (resize to 300x300, normalize) and postprocessing (NMS, coordinate scaling). Uses Edge TPU delegate when available, falls back to CPU TFLite on laptop.

### tracker/feature_extractor.py
Wraps TFLite interpreter for MobileNetV3-Small. One method: `extract(frame, bbox) -> np.ndarray` (1024-dim feature vector). Crops bbox from frame, resizes to 224x224, runs inference, returns feature vector from penultimate layer. Same Edge TPU / CPU fallback.

### tracker/kalman_filter.py
Standard linear Kalman filter. State: `[x, y, w, h, dx, dy, dw, dh]`. Methods: `predict() -> bbox`, `update(bbox)`, `get_uncertainty() -> float`. Copied from SAMURAI fork's Kalman implementation, stripped of any PyTorch dependency. Pure numpy.

### tracker/data_association.py
One function: `match_detection(detections: list, predicted_bbox: tuple) -> Optional[Detection]`. Computes IoU between each detection and the Kalman prediction. Returns the best match above a minimum IoU threshold, or None if no match.

### tracker/state_machine.py
Copied from SAMURAI fork. Adapted: removed any torch imports, uses plain floats. Interface unchanged: `update(confidence: float) -> State`. Added bbox stability check: rapid area change forces UNCERTAIN.

### tracker/smart_memory.py
Copied from SAMURAI fork. Adapted: stores numpy feature vectors instead of SAM2 memory embeddings. STM is a list of (feature_vector, confidence) tuples. LTM is same. Interface unchanged: `should_write()`, `update_ltm()`, `get_features()`.

### tracker/appearance_recovery.py
Copied from SAMURAI fork. Adapted: operates on numpy arrays. Interface: `store_fingerprint(feature, confidence)`, `search(all_detection_features: list) -> Optional[int]` (returns index of best matching detection, or None).

### pipeline/tracker_pipeline.py
Orchestrates everything. Single class `TrackerPipeline` with:
- `initialize(frame, selected_bbox)` — sets up Kalman, stores initial fingerprint
- `process_frame(frame) -> TrackingResult` — runs the full per-frame loop
- Loads all components, reads config from YAML

### scripts/select_target.py
OpenCV window. Runs detector on first frame. Draws all detected bboxes with class labels. User clicks on one to select it as the target. Returns the selected bbox.

---

## Adaptation Notes (SAMURAI → Edge)

### What Changes

| Component | SAMURAI Fork | Edge Project |
|---|---|---|
| Feature type | torch.Tensor (GPU) | np.ndarray (CPU) |
| Feature source | SAM2 image encoder pix_feat | MobileNetV3-Small output |
| Feature dimension | Varies by SAM2 backbone | 1024 (MobileNetV3-Small) |
| Confidence source | s_mask, s_obj, s_kf combined | SSD detection confidence |
| Tracking mechanism | SAM2 memory attention + mask propagation | Detection every frame + IoU matching |
| Memory entries | SAM2 spatial embeddings + object pointers | MobileNetV3 feature vectors |
| Framework | PyTorch + CUDA | TFLite + numpy |
| Similarity computation | torch.nn.functional.cosine_similarity | numpy dot product / scipy cosine |

### What Stays Identical

| Component | Notes |
|---|---|
| State machine logic | Same states, same transitions, same thresholds |
| STM/LTM freeze rules | Same per-state write behavior |
| Bbox stability check | Same area-change threshold |
| Kalman filter | Same state vector, same predict-correct cycle |
| Cosine matching | Same algorithm, different implementation (numpy vs torch) |
| Visualization | Same color scheme per state |
| Config structure | Same YAML, same parameter names |

---

## Configuration

```yaml
# configs/tracker_params.yaml

detector:
  model_path: "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
  input_size: 300
  confidence_threshold: 0.5       # Minimum detection confidence
  nms_iou_threshold: 0.45         # NMS overlap threshold
  use_edgetpu: true               # false for laptop development

feature_extractor:
  model_path: "models/mobilenet_v3_small_quant_edgetpu.tflite"
  input_size: 224
  feature_dim: 1024
  use_edgetpu: true               # false for laptop development

kalman:
  process_noise: 1.0
  measurement_noise: 1.0

data_association:
  min_iou_threshold: 0.3          # Minimum IoU to match detection to prediction

state_machine:
  tau_reliable: 0.6               # Confidence for TRACKING
  tau_lost: 0.2                   # Confidence for LOST
  tau_uncertain_write: 0.5        # STM write threshold in UNCERTAIN
  k_confirm: 3                    # Consecutive frames to confirm state change
  n_lost_frames: 5                # Low-confidence frames before LOST
  t_max_lost: 150                 # Max frames in LOST (~5s at 30fps)
  bbox_shrink_threshold: 0.4      # Area ratio below which forces UNCERTAIN
  bbox_grow_threshold: 2.5        # Area ratio above which forces UNCERTAIN
  bbox_check_window: 5            # Frames over which to measure bbox change

smart_memory:
  stm_size: 7                     # Short-term memory slots
  ltm_size: 5                     # Long-term memory slots
  ltm_min_confidence: 0.7         # Minimum score for LTM candidate

appearance_recovery:
  tau_reid: 0.6                   # Cosine similarity threshold
  max_candidates: 20              # Max detections to compare against

visualization:
  tracking_color: [0, 255, 0]     # Green solid
  uncertain_color: [255, 255, 0]  # Yellow dashed
  lost_color: [255, 0, 0]         # Red dashed
  recovered_color: [0, 255, 255]  # Cyan solid
  show_all_detections: false      # Draw all detector bboxes (debug mode)
  log_path: "logs/tracking.csv"   # Trajectory log file
```

---

## Code Conventions

- Python >=3.9 (Mendel Linux on Coral ships 3.9)
- No PyTorch, no TensorFlow (full). Only tflite-runtime and numpy.
- Type hints on all function signatures
- Docstrings on all public methods
- No classes where a function suffices
- Configuration from YAML only, never hardcoded thresholds
- Logging via Python logging module
- All tracker/ modules independently testable without models loaded (mock inputs)
- Edge TPU / CPU fallback controlled by single `use_edgetpu` flag in config

---

## Dependencies

```
# requirements.txt
numpy>=1.24
opencv-python>=4.8
pyyaml>=6.0
scipy>=1.10
tflite-runtime>=2.14        # CPU inference on laptop
# On Coral board, also install:
# pycoral>=2.0              # Edge TPU delegate
```

---

## Execution Plan

### Phase 0: Project Setup
**Goal:** Create repo, install dependencies, download models, verify inference runs.

Steps:
1. Create repo with directory structure above
2. Install dependencies: `pip install -r requirements.txt`
3. Download SSD-MobileNetV2 Edge TPU model from Coral model zoo
4. Download MobileNetV3-Small, quantize to INT8 TFLite, compile for Edge TPU (or find pre-compiled)
5. Write `tracker/detector.py` — verify it detects objects in a test image
6. Write `tracker/feature_extractor.py` — verify it produces a 1024-dim vector from a crop
7. Both should work on laptop CPU with `use_edgetpu: false`

### Phase 1: Core Tracking Loop (No Occlusion Logic)
**Goal:** Basic detection-association tracking works end-to-end.

Steps:
1. Write `tracker/kalman_filter.py` (adapt from SAMURAI fork, pure numpy)
2. Write `tracker/data_association.py` (IoU matching)
3. Write `scripts/select_target.py` (user clicks to select object)
4. Write `pipeline/tracker_pipeline.py` — basic loop: detect → match → update Kalman → draw bbox
5. Write `scripts/run_video.py` — run on a sample video
6. Verify: target is tracked with green bbox, Kalman smooths jitter
7. No state machine, no memory, no recovery yet — just detection + Kalman

### Phase 2: State Machine
**Goal:** Add occlusion awareness.

Steps:
1. Copy `state_machine.py` from SAMURAI fork
2. Strip torch imports, adapt to plain float inputs
3. Add bbox stability check (area change threshold)
4. Wire into `tracker_pipeline.py`: after each frame, call `state_machine.update(confidence)`
5. Update `visualization/display.py`: bbox color changes by state
6. Write `tests/test_state_machine.py` — same test cases as SAMURAI fork
7. Test on video with occlusion: verify state transitions, verify bbox colors change

### Phase 3: Smart Memory
**Goal:** Prevent feature pollution during occlusion.

Steps:
1. Copy `smart_memory.py` from SAMURAI fork
2. Adapt: store numpy feature vectors instead of SAM2 embeddings
3. Wire into pipeline: store features in STM/LTM based on state
4. Write `tests/test_smart_memory.py`
5. At this point, memory isn't used for anything yet — it stores features that Phase 4 will use

### Phase 4: Appearance Recovery
**Goal:** Re-find target after occlusion.

Steps:
1. Copy `appearance_recovery.py` from SAMURAI fork
2. Adapt: numpy cosine similarity instead of torch
3. Wire into pipeline: when LOST, extract features from ALL detections in frame, compare against stored fingerprints, best match above threshold → RECOVERED
4. Write `tests/test_appearance_recovery.py`
5. Test on video with long occlusion: object disappears behind obstacle, reappears, tracker re-locks
6. Tune tau_reid threshold

### Phase 5: Deploy to Coral
**Goal:** Run on real hardware.

Steps:
1. Flash Mendel Linux on Coral Dev Board Mini
2. Install pycoral, tflite-runtime with Edge TPU delegate
3. Copy project to board
4. Set `use_edgetpu: true` in config
5. Run on recorded video — measure FPS
6. Connect CSI camera — run live
7. Profile: identify bottleneck (TPU inference? CPU logic? frame capture?)
8. Optimize if below 15 FPS

### Phase 6: Polish
**Goal:** Clean up for demo.

Steps:
1. Add trajectory logging to CSV
2. Add FPS counter overlay
3. Add debug mode: draw all detections, show confidence scores, show state label
4. Write README
5. Record demo video

---

## Visualization by State

| State | Bbox Style | Color | Label |
|---|---|---|---|
| TRACKING | Solid, 2px | Green (0, 255, 0) | None |
| UNCERTAIN | Dashed, 2px | Yellow (255, 255, 0) | UNCERTAIN |
| LOST | Dashed, 1px | Red (255, 0, 0) | LOST (Kalman prediction) |
| RECOVERED | Solid, 3px | Cyan (0, 255, 255) | RECOVERED (first 10 frames) |

During LOST, the drawn bbox is the Kalman filter's predicted position. During all other states, it's the matched detection bbox.

---

## Laptop vs Coral: Same Code, Different Speed

```python
# In detector.py and feature_extractor.py:
if config['use_edgetpu']:
    from pycoral.utils.edgetpu import make_interpreter
    interpreter = make_interpreter(model_path)
else:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path)
```

This is the ONLY difference between laptop and Coral. All logic, all thresholds, all pipeline code is identical. Develop fast on laptop, deploy unchanged to board.

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| MobileNetV3 features not discriminative enough | Fall back to simpler template matching (NCC on raw pixel crops) |
| SSD-MobileNetV2 misses small objects | Lower confidence threshold, accept more false positives filtered by IoU matching |
| 2GB RAM too tight | Profile memory, reduce frame buffer size, use 320x240 capture if needed |
| Edge TPU model compilation fails | Use pre-compiled models from Coral model zoo |
| FPS below 15 on Coral | Run detector at lower resolution (300→240), skip feature extraction when TRACKING is stable |
| Object not in COCO 80 classes | Known limitation of Approach A. Document clearly. Future: Approach B (hybrid) |
