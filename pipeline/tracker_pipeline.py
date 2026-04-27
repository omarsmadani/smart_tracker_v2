"""Phase 4 pipeline: detect -> IoU-match -> Kalman -> state machine -> memory -> recovery."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from tracker.appearance_recovery import AppearanceRecovery
from tracker.data_association import match_detection
from tracker.detector import Detection, Detector
from tracker.feature_extractor import FeatureExtractor
from tracker.kalman_filter import KalmanBoxFilter
from tracker.smart_memory import SmartMemory
from tracker.state_machine import State, StateMachine

Bbox = Tuple[int, int, int, int]


@dataclass
class TrackingResult:
    bbox: Bbox
    state: State
    matched: bool
    confidence: float
    class_id: int
    raw_detection: Optional[Detection]
    recovered_frames: int = 0


class TrackerPipeline:
    def __init__(self, config: dict):
        self.detector = Detector(config["detector"])
        self.feature_extractor = FeatureExtractor(config["feature_extractor"])
        self.kalman = KalmanBoxFilter(
            process_noise=config["kalman"]["process_noise"],
            measurement_noise=config["kalman"]["measurement_noise"],
        )
        self.state_machine = StateMachine(config["state_machine"])
        self.memory = SmartMemory(config["smart_memory"])
        self.recovery = AppearanceRecovery(config["appearance_recovery"])
        self.min_iou: float = config["data_association"]["min_iou_threshold"]
        self.tau_uncertain_write: float = config["state_machine"]["tau_uncertain_write"]
        self.target_class: int = -1
        self._recovered_frames: int = 0

    def initialize(self, frame: np.ndarray, bbox: Bbox, class_id: int) -> None:
        self.kalman.initialize(tuple(float(v) for v in bbox))
        self.target_class = int(class_id)
        self.state_machine.state = State.TRACKING
        self._recovered_frames = 0

        feat = self._safe_extract(frame, bbox)
        if feat is not None:
            self.memory.update(feat, 1.0, State.TRACKING)

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        predicted = self.kalman.predict()
        detections = self.detector.detect(frame)

        # ── LOST: try appearance recovery before IoU match ──────────
        if self.state_machine.state == State.LOST:
            result = self._attempt_recovery(frame, detections, predicted)
            if result is not None:
                return result

        # ── Normal: IoU association ──────────────────────────────────
        match = match_detection(detections, predicted, self.min_iou, self.target_class)
        if match is None:
            match = match_detection(detections, predicted, self.min_iou, None)

        if match is not None:
            updated = self.kalman.update(tuple(float(v) for v in match.bbox))
            bbox_draw = _to_int_bbox(updated)
            confidence = match.confidence
        else:
            bbox_draw = _to_int_bbox(predicted)
            confidence = 0.0

        state = self.state_machine.update(confidence, bbox_draw)

        if state == State.RECOVERED:
            self._recovered_frames += 1
        else:
            self._recovered_frames = 0

        # ── Memory writes ────────────────────────────────────────────
        if match is not None:
            feat = self._safe_extract(frame, match.bbox)
            if feat is not None:
                if state == State.UNCERTAIN:
                    self.memory.update_uncertain(feat, confidence, self.tau_uncertain_write)
                elif state != State.LOST:
                    self.memory.update(feat, confidence, state)

        # Refresh fingerprint whenever memory is written (not during LOST)
        if state != State.LOST:
            self.recovery.store_fingerprint(self.memory.get_features())

        return TrackingResult(
            bbox=bbox_draw,
            state=state,
            matched=match is not None,
            confidence=confidence,
            class_id=match.class_id if match else self.target_class,
            raw_detection=match,
            recovered_frames=self._recovered_frames,
        )

    # ── recovery ────────────────────────────────────────────────────

    def _attempt_recovery(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        predicted: tuple,
    ) -> Optional[TrackingResult]:
        if not detections:
            self.state_machine.update(0.0, _to_int_bbox(predicted))
            return TrackingResult(
                bbox=_to_int_bbox(predicted), state=State.LOST,
                matched=False, confidence=0.0,
                class_id=self.target_class, raw_detection=None,
            )

        cand_features = [self._safe_extract(frame, d.bbox) for d in detections]
        valid = [(i, f) for i, f in enumerate(cand_features) if f is not None]

        best_idx = self.recovery.search([f for _, f in valid])

        if best_idx is not None:
            det_idx = valid[best_idx][0]
            match = detections[det_idx]
            self.kalman.update(tuple(float(v) for v in match.bbox))
            self.state_machine.force_recovered()
            self._recovered_frames = 1

            feat = valid[best_idx][1]
            self.memory.update(feat, match.confidence, State.RECOVERED)
            self.recovery.store_fingerprint(self.memory.get_features())

            return TrackingResult(
                bbox=_to_int_bbox(match.bbox),
                state=State.RECOVERED,
                matched=True,
                confidence=match.confidence,
                class_id=match.class_id,
                raw_detection=match,
                recovered_frames=self._recovered_frames,
            )

        # No re-id match — advance LOST counter normally
        self.state_machine.update(0.0, _to_int_bbox(predicted))
        return None  # fall through to normal path (draws Kalman prediction)

    def _safe_extract(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        try:
            return self.feature_extractor.extract(frame, bbox)
        except Exception:
            return None


def _to_int_bbox(bbox) -> Bbox:
    x, y, w, h = bbox
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))
