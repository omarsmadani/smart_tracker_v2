"""Thin CSRT wrapper used as a bootstrap fallback before detection locks on."""
from typing import Optional, Tuple

import cv2
import numpy as np

Bbox = Tuple[int, int, int, int]


def _make_csrt():
    # Handle namespace differences across OpenCV builds
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise ImportError(
        "CSRT tracker not found. Install opencv-contrib-python:\n"
        "  pip install opencv-contrib-python"
    )


class TemplateTracker:
    def __init__(self):
        self._tracker = None
        self._active = False

    def init(self, frame: np.ndarray, bbox: Bbox) -> None:
        self._tracker = _make_csrt()
        self._tracker.init(frame, tuple(int(v) for v in bbox))
        self._active = True

    def update(self, frame: np.ndarray) -> Optional[Bbox]:
        """Returns updated bbox, or None if tracking was lost."""
        if not self._active or self._tracker is None:
            return None
        ok, box = self._tracker.update(frame)
        if not ok:
            self._active = False
            return None
        x, y, w, h = box
        return (int(x), int(y), int(w), int(h))

    def reset(self) -> None:
        self._tracker = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active
