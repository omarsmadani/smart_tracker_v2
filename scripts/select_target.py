"""Target selection UI: user click-and-drags a ROI box over any object."""
from typing import Optional, Tuple

import cv2
import numpy as np

Bbox = Tuple[int, int, int, int]

_INSTRUCTIONS = "Drag a box around the object to track  |  C = cancel"


def select(
    frame: np.ndarray,
    cap: Optional[cv2.VideoCapture] = None,
) -> Optional[Bbox]:
    """Show a ROI-drag selection window.

    For camera mode pass `cap` so the user picks from a live frame.
    Returns (x, y, w, h) in pixel coordinates, or None if cancelled.
    """
    # For camera: grab a fresh frame so the image isn't stale
    if cap is not None and cap.isOpened():
        ok, live = cap.read()
        if ok:
            frame = live

    # Draw instruction banner onto a copy so the original stays clean
    display = frame.copy()
    h = display.shape[0]
    (tw, th), _ = cv2.getTextSize(_INSTRUCTIONS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(display, (0, h - th - 16), (display.shape[1], h), (0, 0, 0), -1)
    cx = (display.shape[1] - tw) // 2
    cv2.putText(display, _INSTRUCTIONS, (cx, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # selectROI: returns (x, y, w, h); pressing C or Esc returns (0,0,0,0)
    roi = cv2.selectROI("Edge Tracker — Select Target", display,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Edge Tracker — Select Target")

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (int(x), int(y), int(w), int(h))
