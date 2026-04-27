"""Cosine-similarity appearance recovery. Adapted from SAMURAI fork — numpy, no torch."""
from typing import List, Optional

import numpy as np


class AppearanceRecovery:
    def __init__(self, config: dict):
        self.tau_reid: float = config["tau_reid"]
        self.max_candidates: int = config["max_candidates"]
        self._fingerprint: List[np.ndarray] = []

    def store_fingerprint(self, features: List[np.ndarray]) -> None:
        """Replace fingerprint with the current memory snapshot."""
        self._fingerprint = [f for f in features if f is not None and f.size > 0]

    def search(self, candidate_features: List[np.ndarray]) -> Optional[int]:
        """Compare each candidate against stored fingerprint.

        Returns index into candidate_features of the best match above tau_reid,
        or None if no match clears the threshold.
        """
        if not self._fingerprint or not candidate_features:
            return None

        candidates = candidate_features[: self.max_candidates]
        best_idx: Optional[int] = None
        best_score = self.tau_reid  # must beat threshold to count

        for i, cand in enumerate(candidates):
            score = self._mean_cosine(cand)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _mean_cosine(self, candidate: np.ndarray) -> float:
        """Mean cosine similarity between candidate and all fingerprint vectors.

        Both sides are assumed L2-normalised, so cosine = dot product.
        """
        scores = [float(np.dot(candidate, fp)) for fp in self._fingerprint]
        return float(np.mean(scores)) if scores else 0.0
