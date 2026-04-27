"""STM + LTM feature memory. Adapted from SAMURAI fork — numpy arrays, no torch."""
from typing import List, NamedTuple, Optional

import numpy as np

from .state_machine import State


class MemoryEntry(NamedTuple):
    feature: np.ndarray   # L2-normalised 1-D vector
    confidence: float


class SmartMemory:
    def __init__(self, config: dict):
        self.stm_size: int = config["stm_size"]
        self.ltm_size: int = config["ltm_size"]
        self.ltm_min_conf: float = config["ltm_min_confidence"]

        self._stm: List[MemoryEntry] = []
        self._ltm: List[MemoryEntry] = []

    # ── write policy ────────────────────────────────────────────────

    def update(self, feature: np.ndarray, confidence: float, state: State) -> None:
        """Write feature into STM/LTM according to per-state rules."""
        if state == State.TRACKING:
            self._write_stm(feature, confidence)
            self._try_update_ltm(feature, confidence)

        elif state == State.UNCERTAIN:
            # STM write threshold enforced by caller via update_uncertain()
            pass
            # LTM frozen in UNCERTAIN

        elif state == State.LOST:
            pass  # no writes during LOST

        elif state == State.RECOVERED:
            self._write_stm(feature, confidence)
            # LTM kept as-is (identity preserved across occlusion)

    def update_uncertain(self, feature: np.ndarray, confidence: float,
                         tau_uncertain_write: float) -> None:
        """Separate entry-point for UNCERTAIN to enforce write threshold."""
        if confidence >= tau_uncertain_write:
            self._write_stm(feature, confidence)

    # ── read ────────────────────────────────────────────────────────

    def get_features(self) -> List[np.ndarray]:
        """Return all stored features (STM + LTM). Used by appearance recovery."""
        return [e.feature for e in self._stm] + [e.feature for e in self._ltm]

    def get_ltm_features(self) -> List[np.ndarray]:
        return [e.feature for e in self._ltm]

    def get_stm_features(self) -> List[np.ndarray]:
        return [e.feature for e in self._stm]

    # ── internals ───────────────────────────────────────────────────

    def _write_stm(self, feature: np.ndarray, confidence: float) -> None:
        self._stm.append(MemoryEntry(feature, confidence))
        if len(self._stm) > self.stm_size:
            self._stm.pop(0)

    def _try_update_ltm(self, feature: np.ndarray, confidence: float) -> None:
        if confidence < self.ltm_min_conf:
            return
        if len(self._ltm) < self.ltm_size:
            self._ltm.append(MemoryEntry(feature, confidence))
            self._ltm.sort(key=lambda e: e.confidence)
            return
        # Replace the lowest-confidence LTM entry if this is better
        if confidence > self._ltm[0].confidence:
            self._ltm[0] = MemoryEntry(feature, confidence)
            self._ltm.sort(key=lambda e: e.confidence)

    @property
    def stm_count(self) -> int:
        return len(self._stm)

    @property
    def ltm_count(self) -> int:
        return len(self._ltm)
