"""CSRT-primary pipeline: arbitrary object tracking, template-search recovery."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracker.appearance_recovery import AppearanceRecovery
from tracker.feature_extractor import FeatureExtractor
from tracker.kalman_filter import KalmanBoxFilter
from tracker.smart_memory import SmartMemory
from tracker.state_machine import State, StateMachine
from tracker.template_tracker import TemplateTracker
import tracker.template_search as template_search

Bbox = Tuple[int, int, int, int]


@dataclass
class TrackingResult:
    bbox: Bbox
    state: State
    matched: bool
    confidence: float
    recovered_frames: int = 0


class TrackerPipeline:
    def __init__(self, config: dict):
        self.feature_extractor = FeatureExtractor(config["feature_extractor"])
        self.kalman = KalmanBoxFilter(
            process_noise=config["kalman"]["process_noise"],
            measurement_noise=config["kalman"]["measurement_noise"],
        )
        self.state_machine = StateMachine(config["state_machine"])
        self.memory = SmartMemory(config["smart_memory"])
        self.recovery = AppearanceRecovery(config["appearance_recovery"])
        self.template = TemplateTracker()

        ts = config["template_search"]
        self._ts_scales: List[float] = ts["scales"]
        self._ts_top_k: int = ts["top_k"]
        self._ts_method: int = getattr(cv2, ts.get("match_method", "TM_CCOEFF_NORMED"))
        self._ts_nms_ratio: float = ts["nms_distance_ratio"]

        dc = config["drift_check"]
        self._drift_interval: int = dc["interval_frames"]
        self._drift_min_drop: float = dc["min_cosine_drop"]

        self.tau_uncertain_write: float = config["state_machine"]["tau_uncertain_write"]
        self.tau_reid: float = config["appearance_recovery"]["tau_reid"]

        self._seed_crop: Optional[np.ndarray] = None
        self._cos_peak: float = 0.0
        self._frames_since_drift: int = 0
        self._recovered_frames: int = 0

    def initialize(self, frame: np.ndarray, bbox: Bbox) -> None:
        x, y, w, h = bbox
        self._seed_crop = frame[y: y + h, x: x + w].copy()

        self.kalman.initialize(tuple(float(v) for v in bbox))
        self.state_machine.state = State.TRACKING
        self._cos_peak = 0.0
        self._frames_since_drift = 0
        self._recovered_frames = 0

        self.template.reset()
        self.template.init(frame, bbox)

        feat = self._safe_extract(frame, bbox)
        if feat is not None:
            self.memory.update(feat, 1.0, State.TRACKING)
            self.recovery.store_fingerprint(self.memory.get_features())
            self._cos_peak = 1.0

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        predicted = self.kalman.predict()

        # ── LOST: attempt template-search recovery ───────────────────
        if self.state_machine.state == State.LOST:
            result = self._attempt_recovery(frame)
            if result is not None:
                return result
            # Recovery failed — return ghost bbox
            self.state_machine.update(0.0, _to_int_bbox(predicted))
            return TrackingResult(
                bbox=_to_int_bbox(predicted),
                state=State.LOST,
                matched=False,
                confidence=0.0,
            )

        # ── Primary: CSRT update ─────────────────────────────────────
        csrt_bbox = self.template.update(frame)

        if csrt_bbox is None:
            bbox = _to_int_bbox(predicted)
            confidence = 0.0
            feat = None
        else:
            self.kalman.update(tuple(float(v) for v in csrt_bbox))
            bbox = csrt_bbox
            feat = self._safe_extract(frame, bbox)
            confidence = self.recovery.mean_cosine(feat) if feat is not None else 0.0

        self._cos_peak = max(self._cos_peak, confidence)
        state = self.state_machine.update(confidence, bbox)

        if state == State.RECOVERED:
            self._recovered_frames += 1
        else:
            self._recovered_frames = 0

        # ── Memory writes ────────────────────────────────────────────
        if feat is not None:
            if state == State.UNCERTAIN:
                self.memory.update_uncertain(feat, confidence, self.tau_uncertain_write)
            elif state != State.LOST:
                self.memory.update(feat, confidence, state)
            if state != State.LOST:
                self.recovery.store_fingerprint(self.memory.get_features())

        # ── Drift check (TRACKING only) ──────────────────────────────
        if state == State.TRACKING:
            self._frames_since_drift += 1
            if (self._frames_since_drift >= self._drift_interval and
                    self._cos_peak - confidence > self._drift_min_drop):
                self._correct_drift(frame)

        return TrackingResult(
            bbox=bbox,
            state=state,
            matched=csrt_bbox is not None,
            confidence=confidence,
            recovered_frames=self._recovered_frames,
        )

    # ── recovery ─────────────────────────────────────────────────────

    def _attempt_recovery(self, frame: np.ndarray) -> Optional[TrackingResult]:
        if self._seed_crop is None:
            return None

        candidates = template_search.search(
            frame, self._seed_crop,
            self._ts_scales, self._ts_top_k,
            self._ts_method, self._ts_nms_ratio,
        )
        if not candidates:
            return None

        feats = [self._safe_extract(frame, bbox) for bbox, _ in candidates]
        valid = [(bbox, f) for (bbox, _), f in zip(candidates, feats) if f is not None]
        if not valid:
            return None

        best_idx = self.recovery.search([f for _, f in valid])
        if best_idx is None:
            return None

        recovered_bbox, best_feat = valid[best_idx]
        self.template.reset()
        self.template.init(frame, recovered_bbox)
        self.kalman.update(tuple(float(v) for v in recovered_bbox))
        self.state_machine.force_recovered()
        self._recovered_frames = 1
        self._cos_peak = self.recovery.mean_cosine(best_feat)
        self._frames_since_drift = 0

        self.memory.update(best_feat, self._cos_peak, State.RECOVERED)
        self.recovery.store_fingerprint(self.memory.get_features())

        return TrackingResult(
            bbox=recovered_bbox,
            state=State.RECOVERED,
            matched=True,
            confidence=self._cos_peak,
            recovered_frames=self._recovered_frames,
        )

    def _correct_drift(self, frame: np.ndarray) -> bool:
        self._frames_since_drift = 0
        if self._seed_crop is None:
            return False

        candidates = template_search.search(
            frame, self._seed_crop,
            self._ts_scales, self._ts_top_k,
            self._ts_method, self._ts_nms_ratio,
        )
        if not candidates:
            return False

        feats = [self._safe_extract(frame, bbox) for bbox, _ in candidates]
        best_score = self.tau_reid
        best_bbox = None
        best_feat = None

        for (bbox, _), feat in zip(candidates, feats):
            if feat is None:
                continue
            score = self.recovery.mean_cosine(feat)
            if score > best_score:
                best_score = score
                best_bbox = bbox
                best_feat = feat

        if best_bbox is None:
            return False

        self.template.reset()
        self.template.init(frame, best_bbox)
        self._cos_peak = best_score
        if best_feat is not None:
            self.memory.update(best_feat, best_score, State.TRACKING)
            self.recovery.store_fingerprint(self.memory.get_features())
        return True

    def _safe_extract(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        try:
            return self.feature_extractor.extract(frame, bbox)
        except Exception:
            return None


def _to_int_bbox(bbox) -> Bbox:
    x, y, w, h = bbox
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))
