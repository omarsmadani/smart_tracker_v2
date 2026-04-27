"""Occlusion state machine. Adapted from SAMURAI fork — torch removed, plain floats."""
from collections import deque
from enum import Enum
from typing import Optional


class State(Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    RECOVERED = "RECOVERED"


class StateMachine:
    def __init__(self, config: dict):
        self.tau_reliable: float = config["tau_reliable"]
        self.tau_lost: float = config["tau_lost"]
        self.tau_uncertain_write: float = config["tau_uncertain_write"]
        self.k_confirm: int = config["k_confirm"]
        self.n_lost_frames: int = config["n_lost_frames"]
        self.t_max_lost: int = config["t_max_lost"]
        self.bbox_shrink: float = config["bbox_shrink_threshold"]
        self.bbox_grow: float = config["bbox_grow_threshold"]
        self.bbox_window: int = config["bbox_check_window"]

        self.state = State.TRACKING
        self._low_conf_streak: int = 0
        self._high_conf_streak: int = 0
        self._lost_frames: int = 0
        self._recovered_frames: int = 0
        self._area_history: deque = deque(maxlen=self.bbox_window)

    # ------------------------------------------------------------------
    def update(self, confidence: float, bbox: Optional[tuple] = None) -> State:
        """Advance the state machine one frame.

        Args:
            confidence: detector confidence for this frame (0.0 if no match).
            bbox: current (x, y, w, h) for bbox stability check; None skips check.
        Returns:
            New State.
        """
        bbox_unstable = self._check_bbox_stability(bbox)

        if self.state == State.TRACKING:
            self.state = self._from_tracking(confidence, bbox_unstable)

        elif self.state == State.UNCERTAIN:
            self.state = self._from_uncertain(confidence, bbox_unstable)

        elif self.state == State.LOST:
            self.state = self._from_lost(confidence)

        elif self.state == State.RECOVERED:
            self.state = self._from_recovered(confidence)

        return self.state

    def force_recovered(self) -> None:
        """Called by appearance recovery when a re-id match is found."""
        self.state = State.RECOVERED
        self._recovered_frames = 0
        self._lost_frames = 0
        self._low_conf_streak = 0

    # ------------------------------------------------------------------
    def _from_tracking(self, conf: float, unstable: bool) -> State:
        if unstable or conf < self.tau_reliable:
            self._low_conf_streak += 1
            self._high_conf_streak = 0
        else:
            self._low_conf_streak = 0
            self._high_conf_streak += 1

        if conf < self.tau_lost:
            if self._low_conf_streak >= self.n_lost_frames:
                self._lost_frames = 0
                return State.LOST
        if unstable or conf < self.tau_reliable:
            if self._low_conf_streak >= self.k_confirm:
                return State.UNCERTAIN
        return State.TRACKING

    def _from_uncertain(self, conf: float, unstable: bool) -> State:
        if conf >= self.tau_reliable and not unstable:
            self._high_conf_streak += 1
            self._low_conf_streak = 0
        else:
            self._low_conf_streak += 1
            self._high_conf_streak = 0

        if conf < self.tau_lost:
            if self._low_conf_streak >= self.n_lost_frames:
                self._lost_frames = 0
                return State.LOST
        if self._high_conf_streak >= self.k_confirm:
            return State.TRACKING
        return State.UNCERTAIN

    def _from_lost(self, conf: float) -> State:
        self._lost_frames += 1
        if self._lost_frames >= self.t_max_lost:
            # Give up — stay LOST; pipeline can decide to terminate
            return State.LOST
        # Recovery is triggered externally via force_recovered()
        # A spontaneous high-conf detection also recovers (no memory yet)
        if conf >= self.tau_reliable:
            self._lost_frames = 0
            self._recovered_frames = 0
            return State.RECOVERED
        return State.LOST

    def _from_recovered(self, conf: float) -> State:
        self._recovered_frames += 1
        if conf < self.tau_lost:
            self._lost_frames = 0
            return State.LOST
        if self._recovered_frames >= self.k_confirm:
            return State.TRACKING
        return State.RECOVERED

    def _check_bbox_stability(self, bbox: Optional[tuple]) -> bool:
        if bbox is None:
            return False
        x, y, w, h = bbox
        area = float(w * h)
        self._area_history.append(area)
        if len(self._area_history) < self.bbox_window:
            return False
        ref = self._area_history[0]
        if ref <= 0:
            return False
        ratio = area / ref
        return ratio < self.bbox_shrink or ratio > self.bbox_grow
