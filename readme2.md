# Smart Tracker — Architecture and Progress

## 1. Purpose

Smart Tracker is a single-object visual tracker designed to run on the Coral Dev Board Mini (4 TOPS Edge TPU, 2 GB RAM). It tracks one user-selected object through a video stream and is designed to keep its identity across short and medium-length occlusions.

The tracker follows a detection-association paradigm: a detector runs every frame, and the target is identified among the detections using motion (Kalman prediction + IoU) and appearance (deep feature cosine similarity). An occlusion-aware state machine, a two-tier feature memory, and a re-identification recovery routine sit on top of the per-frame loop to handle disappearances without drifting onto distractors.

This document describes the architecture as currently implemented, the components inherited from the SAMURAI fork, the current implementation state, the planned remaining phases, and a recommended benchmark plan.

---

## 2. High-Level Architecture

The tracker is a per-frame pipeline composed of swappable modules. One frame produces one tracking decision.

```
                                ┌────────────────────────┐
   frame ─────────────────────► │ Detector (SSD-MNetV2)  │ ─► detections
                                └────────────────────────┘
                                            │
                                            ▼
                            ┌─────────────────────────────────┐
                            │ Feature Extractor (MNetV2)      │ ─► per-detection feature
                            └─────────────────────────────────┘
                                            │
                                            ▼
   Kalman prediction ─►  ┌──────────────────────────────────────┐
   memory fingerprints ─►│ Data Association (IoU + cosine)      │ ─► best detection or none
                          └──────────────────────────────────────┘
                                            │
                              match? ┌──────┴───────┐ no match
                                     ▼              ▼
                         ┌────────────────┐   ┌──────────────────────────┐
                         │ Kalman update  │   │ Template fallback (CSRT) │ (bootstrap only)
                         └────────────────┘   └──────────────────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ State Machine          │ ─► TRACKING / UNCERTAIN /
                                │  (confidence + bbox)   │     LOST / RECOVERED
                                └────────────────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Smart Memory (STM/LTM) │ ◄── per-state write rules
                                └────────────────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Appearance Recovery    │ ◄── activated when LOST
                                └────────────────────────┘
```

When the state is LOST, the recovery branch runs first: every detection in the frame is scored against the stored fingerprint, and the best match above a cosine threshold forces the state machine into RECOVERED.

---

## 3. Components

### 3.1 Detector ([tracker/detector.py](tracker/detector.py))

- Wraps a TFLite interpreter for SSD-MobileNetV2 trained on COCO 80 classes.
- Returns `Detection(bbox, class_id, confidence)` for every detection above a confidence floor.
- Same code path runs on Edge TPU (via `pycoral`) and on a laptop CPU (via `tflite-runtime`); the choice is controlled by a single `use_edgetpu` flag in the config.

The pipeline is class-agnostic: `class_id` is recorded for logging only and is not used for association.

### 3.2 Feature Extractor ([tracker/feature_extractor.py](tracker/feature_extractor.py))

- Wraps a TFLite interpreter for a quantized MobileNetV2 classifier.
- Crops the frame to the target bbox, resizes to 224×224, runs inference, and returns the L2-normalized output of the classification head as a 1001-dimensional appearance embedding.
- Note: the codebase uses MobileNetV2 (not MobileNetV3-Small as originally planned). The 1001-dim vector is the classifier logits used as a feature embedding.

### 3.3 Kalman Filter ([tracker/kalman_filter.py](tracker/kalman_filter.py))

- Linear Kalman filter over the bounding-box state `[x, y, w, h, dx, dy, dw, dh]`.
- Pure NumPy implementation; no Torch dependency.
- Predicts the bbox each frame before association; updated with the matched detection bbox after association.

### 3.4 Data Association ([tracker/data_association.py](tracker/data_association.py))

The association rule is a combined motion + appearance score:

```
combined = alpha * IoU(det, predicted) + beta * mean_cosine(det_feat, fingerprints)
```

A candidate detection is only considered if it clears three independent floors:

| Floor | Default | Purpose |
|---|---|---|
| `min_iou_threshold` | 0.2 | Minimum spatial overlap with the Kalman prediction |
| `min_cosine_threshold` | 0.4 | Minimum appearance similarity to memory fingerprints |
| `min_combined_score` | 0.45 | Minimum joint score |

Among candidates that clear all three floors, the highest combined score wins. This rejects nearby distractors that are spatially close but visually different, and conversely rejects look-alikes that are far from the predicted location.

### 3.5 State Machine ([tracker/state_machine.py](tracker/state_machine.py))

Adapted from the SAMURAI occlusion fork. Four states drive memory and recovery behavior:

| State | Meaning | Behavior |
|---|---|---|
| TRACKING | Target reliably observed | Memory writes enabled; LTM may update |
| UNCERTAIN | Confidence dropping or bbox unstable | Selective STM writes only; LTM frozen |
| LOST | No match for `n_lost_frames` consecutive frames | Memory frozen; recovery loop runs |
| RECOVERED | Re-id match found, on probation | STM writes resume; LTM kept as-is |

Transitions require `k_confirm` consecutive frames to debounce noise. A bbox stability check forces UNCERTAIN if the bounding-box area changes by more than the shrink/grow ratios over a sliding window — this catches cases where the detector locks onto a partial occlusion or fuses the target with a neighbor.

The state machine does not match detections itself; it only observes the per-frame confidence and bbox. The pipeline is the only writer of `force_recovered()`, called by the appearance recovery routine.

### 3.6 Smart Memory ([tracker/smart_memory.py](tracker/smart_memory.py))

Two-tier feature buffer adapted from the SAMURAI fork.

- **Short-Term Memory (STM):** FIFO of the most recent N feature vectors (default 7). Captures recent appearance changes (lighting, pose).
- **Long-Term Memory (LTM):** Min-heap of the highest-confidence feature vectors seen so far (default 5, requires confidence ≥ 0.7). Captures stable identity.

Per-state write rules:

| State | STM | LTM |
|---|---|---|
| TRACKING | Write every frame | Replace lowest-confidence entry if current confidence is higher |
| UNCERTAIN | Write only if `confidence ≥ tau_uncertain_write` | Frozen |
| LOST | Frozen | Frozen |
| RECOVERED | Write every frame | Frozen (preserves pre-occlusion identity) |

Freezing LTM during occlusion is the key invariant: it prevents the memory from absorbing features of the occluder.

### 3.7 Appearance Recovery ([tracker/appearance_recovery.py](tracker/appearance_recovery.py))

When the state is LOST, the pipeline:

1. Extracts features for every detection in the current frame (capped at `max_candidates`).
2. Computes the mean cosine similarity between each candidate and the stored fingerprint (the union of STM and LTM at the moment of loss).
3. Returns the index of the best candidate above `tau_reid` (default 0.6), or None.

If a match is found, the pipeline resets the Kalman filter to the recovered bbox and forces the state to RECOVERED. If not, the state stays LOST and the Kalman prediction is drawn instead.

### 3.8 Template Tracker (CSRT) Bootstrap ([tracker/template_tracker.py](tracker/template_tracker.py))

A short-lived OpenCV CSRT tracker used only as a bootstrap fallback in the first frames after target selection. If the detector fails to lock onto the target in the first `bootstrap_frames` frames (default 15), the CSRT tracker provides a bbox so the Kalman filter and memory have something to seed with. Once the detector matches the target once, the CSRT tracker is discarded and never re-engaged.

This decouples target selection from detector recall: the user can select an object even on a frame where the detector temporarily missed it.

### 3.9 Pipeline ([pipeline/tracker_pipeline.py](pipeline/tracker_pipeline.py))

Holds and orchestrates all the above. Two public methods:

- `initialize(frame, bbox)` — seeds Kalman, writes the first feature into memory, stores the initial fingerprint, and primes the CSRT bootstrap.
- `process_frame(frame) -> TrackingResult` — runs the full per-frame loop and returns the chosen bbox, current state, confidence, and the matched raw detection.

---

## 4. Relationship to SAMURAI

The algorithmic layer (state machine, two-tier memory, appearance recovery) is taken from the SAMURAI occlusion fork and adapted to a non-PyTorch environment. The tracking backbone is replaced.

| Component | SAMURAI Fork | Smart Tracker |
|---|---|---|
| Tracking mechanism | SAM2 mask propagation via memory attention | Per-frame detection + IoU/cosine matching |
| Feature source | SAM2 image encoder embeddings | MobileNetV2 classifier logits |
| Feature type | `torch.Tensor` on GPU | `np.ndarray` on CPU |
| Confidence source | Combined `s_mask`, `s_obj`, `s_kf` | Detector confidence |
| Framework | PyTorch + CUDA | TFLite + NumPy |
| State machine | Same four states, same transitions | Same four states, same transitions |
| STM/LTM rules | Same per-state write/freeze policy | Same per-state write/freeze policy |
| Recovery | Cosine matching against fingerprint | Cosine matching against fingerprint |

The intent is to keep the occlusion-handling logic identical in spirit while replacing one heavy generative model (SAM2) with two small discriminative models (SSD-MobileNetV2 + MobileNetV2 features), both runnable on the Edge TPU.

---

## 5. Comparison to Common Tracker Families

This is a generic comparison against well-known tracker families, not against specific implementations.

| Family | Mechanism | Trade-off vs. Smart Tracker |
|---|---|---|
| Correlation-filter trackers (e.g., CSRT, KCF) | Template correlation in image space | Fast on CPU, but drift under occlusion and no class awareness. Smart Tracker uses CSRT only as a bootstrap fallback. |
| Siamese trackers | Cross-correlation between a template and a search region using a deep network | Strong appearance model but typically class-agnostic, no detection prior, and most variants need a GPU. |
| Detection-based multi-object trackers (SORT family) | Detector + Kalman + Hungarian assignment, often with appearance | Same architectural skeleton as Smart Tracker but tuned for many objects, usually without an explicit occlusion state machine or memory freeze. |
| Transformer / mask-propagation trackers (e.g., the SAMURAI lineage) | One large model propagates state forward via attention | High accuracy and strong occlusion handling, but require a GPU and are not deployable on Edge TPU at real-time. |

Smart Tracker positions itself in the detection-based family but adds the occlusion-aware state machine, two-tier memory, and re-identification routine borrowed from the SAMURAI lineage.

---

## 6. Configuration

All thresholds live in [configs/tracker_params.yaml](configs/tracker_params.yaml). No thresholds are hardcoded in source files. The same config is used on laptop and Coral; the only field expected to change between targets is `use_edgetpu`.

Key parameter groups:

- `detector` — model path, input size, confidence floor, NMS IoU.
- `feature_extractor` — model path, input size, feature dimension.
- `kalman` — process and measurement noise.
- `data_association` — IoU floor, cosine floor, combined score floor, weights.
- `template_fallback` — enable/disable, bootstrap window length.
- `state_machine` — confidence thresholds (`tau_reliable`, `tau_lost`, `tau_uncertain_write`), confirmation length, lost-frame budget, bbox stability ratios.
- `smart_memory` — STM and LTM sizes, LTM minimum confidence.
- `appearance_recovery` — re-id cosine threshold, candidate cap.
- `visualization` — bbox colors per state, log path.

---

## 7. Current State

The repository is being built in phases as described in [claude.md](claude.md). Status as of the current commit:

| Phase | Goal | Status |
|---|---|---|
| Phase 0 | Project setup, model download, detector and feature extractor wrappers verified on CPU | Done |
| Phase 1 | Kalman + IoU association + target selection UI + basic pipeline + video runner | Done |
| Phase 2 | Occlusion state machine wired into pipeline, bbox color reflects state, unit tests | Done |
| Phase 3 | Smart memory (STM + LTM) wired in, per-state write rules, unit tests | Done |
| Phase 4 | Appearance recovery wired in, end-to-end re-identification, unit tests | Done |
| Phase 5 | Deploy to Coral Dev Board Mini, measure FPS, profile bottlenecks | Pending |
| Phase 6 | Polish: trajectory CSV logging, FPS overlay, debug overlay, README, demo video | Pending |

Additions beyond the original plan that are already in the codebase:

- **Combined IoU + cosine association** with three independent floors. The original plan called for IoU-only matching; cosine was added so that cluttered scenes with multiple instances of the same class do not cause identity flips.
- **Class-agnostic association.** The pipeline records `class_id` for logging but does not require detections to share the target's class. This was needed because the detector occasionally relabels the target across frames.
- **CSRT bootstrap fallback** ([tracker/template_tracker.py](tracker/template_tracker.py)). Active only for the first `bootstrap_frames` frames before the detector first locks. Discarded permanently after the first detection match.

Test coverage is in place for the four algorithmic modules:

- [tests/test_state_machine.py](tests/test_state_machine.py)
- [tests/test_smart_memory.py](tests/test_smart_memory.py)
- [tests/test_appearance_recovery.py](tests/test_appearance_recovery.py)
- [tests/test_data_association.py](tests/test_data_association.py)

The pipeline currently runs on a laptop CPU (`use_edgetpu: false`). Both Edge TPU and CPU TFLite models are present in [models/](models/), so flipping the flag is the only change required for the Coral deploy.

---

## 8. Planned Phases

### 8.1 Phase 5 — Coral Deployment

1. Flash Mendel Linux on the Coral Dev Board Mini.
2. Install `pycoral` and `tflite-runtime` with the Edge TPU delegate.
3. Copy the project to the board.
4. Set `use_edgetpu: true` in the config.
5. Run on a recorded video and measure end-to-end FPS.
6. Connect the CSI camera and run live.
7. Profile to find the bottleneck (TPU inference, CPU logic, frame capture, or memory copies).
8. Optimize as needed to meet the ≥15 FPS target.

Likely optimizations if FPS falls short:
- Skip per-frame feature extraction during stable TRACKING (only extract on UNCERTAIN/RECOVERED or on confidence dips).
- Reduce detector input size from 300×300 to 240×240.
- Cap the number of detections fed to the feature extractor.

### 8.2 Phase 6 — Polish

1. Write per-frame trajectory log to CSV (frame index, bbox, state, confidence).
2. Draw FPS counter on the frame.
3. Debug overlay: all detector boxes, confidence scores, and state label.
4. Write the user-facing README with run instructions for both laptop and Coral.
5. Record a demo video.

---

## 9. Recommended Benchmarks

The benchmarks below are recommended to evaluate the tracker once Phase 5 is complete. They cover three independent axes: standard accuracy, occlusion robustness (the property the SAMURAI logic is meant to deliver), and edge runtime performance.

### 9.1 Standard Single-Object-Tracking Accuracy

Use established short- and long-term SOT benchmarks. These are the de-facto reference points in the tracker literature.

| Benchmark | Why it matters | Metrics |
|---|---|---|
| OTB-100 | Classic short-term SOT benchmark, well-known baseline numbers exist | Success AUC, Precision |
| LaSOT | Long-term, average sequence length ~2500 frames; tests memory and re-detection | Success AUC, Precision, Normalized Precision |
| GOT-10k | Class-agnostic generalization (test classes disjoint from training) | Average Overlap, Success Rate at IoU 0.5 / 0.75 |
| UAV123 | Aerial / small-object scenarios; stresses the detector's small-box recall | Success AUC, Precision |

Note: classes outside COCO 80 will fail at the detector level. This is a known limitation of the current backbone and should be reported, not hidden.

### 9.2 Occlusion-Specific Evaluation

Standard SOT benchmarks measure overall accuracy but do not isolate occlusion behavior. The following are recommended specifically to validate the state machine, smart memory, and appearance recovery.

| Benchmark | Purpose | Metrics |
|---|---|---|
| LaSOT-extension occluded subset | Long-term occlusion in real scenes | Re-detection rate, time-to-recovery |
| OTB-100 OCC attribute subset | Short-occlusion behavior on a small, well-tagged set | Success AUC restricted to OCC attribute |
| TLP (Track Long and Prosper) | Very long sequences with multiple occlusion events | Success AUC, identity consistency |
| Custom synthetic occlusion set | Inject black rectangles, distractor sprites, or random blur over known ground truth in a clean video; vary occlusion length from 5 to 150 frames | Recovery success rate vs. occlusion length, false re-id rate |

The custom synthetic set is recommended because it is the only way to vary occlusion length deterministically and to measure the false re-identification rate (the tracker re-locking onto a similar-looking distractor). This directly stresses the LTM freeze rule and the `tau_reid` threshold.

### 9.3 Distractor / Identity-Switch Evaluation

Smart Tracker's combined IoU + cosine score is designed to resist distractors of the same class. The following benchmarks isolate that property.

| Benchmark | Purpose | Metrics |
|---|---|---|
| VOT-LT (Long-Term) | Includes target absence; stresses re-detection without false positives | F-measure, Pr, Re |
| Custom distractor set | Two visually similar instances of the same class in the same frame (two people, two cars), with the target undergoing occlusion | Identity consistency rate, ID-switch count |

### 9.4 Edge Runtime Benchmarks

These measure whether the tracker is actually deployable, independent of accuracy.

| Test | What is measured |
|---|---|
| End-to-end FPS on Coral, recorded video | Steady-state throughput on a representative sequence |
| End-to-end FPS on Coral, live CSI camera | Real-world throughput including capture and display |
| Per-stage latency profile | Detector ms, feature extractor ms, association ms, state machine ms, recovery ms — to confirm where time is spent |
| Memory footprint | Peak resident memory measured against the 2 GB board budget |
| Power draw (optional) | Sustained current at 5 V over a 5-minute run |
| Thermal stability (optional) | FPS over a 10-minute run, confirming no throttling |

Targets:

- ≥ 15 FPS end-to-end on Coral at 300×300 detector input.
- Peak memory < 500 MB.
- No FPS regression > 10% between minute 1 and minute 10 of a sustained run.

### 9.5 Ablation Studies

Recommended to confirm each algorithmic component contributes:

| Ablation | What it isolates |
|---|---|
| Disable cosine term in association (`beta = 0`) | Value of appearance score in the matching step |
| Disable LTM (`ltm_size = 0`) | Value of long-term identity preservation across occlusion |
| Disable LTM freeze rule (always allow LTM updates) | Value of the freeze invariant; expected to drift onto distractors |
| Disable appearance recovery (skip the LOST branch) | Value of explicit re-identification vs. spontaneous re-detection |
| Disable bbox stability check | Value of the area-ratio guard against partial occlusion fusion |
| Disable CSRT bootstrap | Whether the bootstrap is actually needed in practice |

Each ablation should be run on the custom synthetic occlusion set and on at least one of LaSOT or TLP.

---

## 10. Known Limitations

- Target classes are restricted to COCO 80. Objects outside this set will not be tracked because the detector will not propose them.
- The feature extractor is a classifier, not a metric-learning embedding. Cosine similarity between classifier logits is a usable proxy but is not the strongest possible appearance signal. A re-identification network trained with a metric-learning loss would likely improve `tau_reid` headroom.
- The pipeline is single-object only. Multi-object tracking is out of scope.
- Edge TPU performance has not been measured on hardware as of this writing; the FPS numbers in [claude.md](claude.md) are estimates.

---

## 11. File Index

| Path | Role |
|---|---|
| [pipeline/tracker_pipeline.py](pipeline/tracker_pipeline.py) | Per-frame orchestration |
| [tracker/detector.py](tracker/detector.py) | SSD-MobileNetV2 wrapper |
| [tracker/feature_extractor.py](tracker/feature_extractor.py) | MobileNetV2 feature wrapper |
| [tracker/kalman_filter.py](tracker/kalman_filter.py) | Kalman bbox filter |
| [tracker/data_association.py](tracker/data_association.py) | Combined IoU + cosine matcher |
| [tracker/state_machine.py](tracker/state_machine.py) | Four-state occlusion FSM |
| [tracker/smart_memory.py](tracker/smart_memory.py) | STM + LTM feature buffer |
| [tracker/appearance_recovery.py](tracker/appearance_recovery.py) | Cosine re-identification |
| [tracker/template_tracker.py](tracker/template_tracker.py) | CSRT bootstrap fallback |
| [scripts/select_target.py](scripts/select_target.py) | First-frame target picker |
| [scripts/run_video.py](scripts/run_video.py) | Recorded-video entry point |
| [scripts/download_models.py](scripts/download_models.py) | Model fetcher |
| [scripts/verify_phase0.py](scripts/verify_phase0.py) | Phase 0 sanity check |
| [visualization/display.py](visualization/display.py) | Bbox overlay, state colors |
| [configs/tracker_params.yaml](configs/tracker_params.yaml) | All thresholds and paths |
| [tests/](tests/) | Unit tests for the four algorithmic modules |
