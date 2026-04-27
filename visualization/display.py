"""Draw tracking results onto a frame. Color and style vary by state."""
from typing import Optional, Tuple

import cv2
import numpy as np

from tracker.state_machine import State

Bbox = Tuple[int, int, int, int]
Color = Tuple[int, int, int]

_STATE_STYLE: dict = {
    State.TRACKING:  {"color": (0, 255, 0),   "thickness": 2, "dashed": False},
    State.UNCERTAIN: {"color": (0, 255, 255),  "thickness": 2, "dashed": True},
    State.LOST:      {"color": (0, 0, 255),    "thickness": 1, "dashed": True},
    State.RECOVERED: {"color": (255, 255, 0),  "thickness": 3, "dashed": False},
}


def draw_tracking(frame: np.ndarray, bbox: Bbox, state: State,
                  label: Optional[str] = None, recovered_frames: int = 0) -> None:
    style = _STATE_STYLE[state]
    color: Color = tuple(style["color"])
    thickness: int = style["thickness"]

    x, y, w, h = [int(v) for v in bbox]

    if style["dashed"]:
        _draw_dashed_rect(frame, x, y, x + w, y + h, color, thickness)
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # State label
    state_label = ""
    if state == State.UNCERTAIN:
        state_label = "UNCERTAIN"
    elif state == State.LOST:
        state_label = "LOST"
    elif state == State.RECOVERED and recovered_frames <= 10:
        state_label = "RECOVERED"

    text = " | ".join(filter(None, [label, state_label]))
    if text:
        cv2.putText(frame, text, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, bbox: Bbox, label: str) -> None:
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)
    cv2.putText(frame, label, (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_dashed_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                      color: Color, thickness: int, gap: int = 10) -> None:
    for seg in _dash_segments(x0, y0, x1, y1, gap):
        cv2.line(img, seg[0], seg[1], color, thickness)


def _dash_segments(x0, y0, x1, y1, gap):
    segs = []
    for (ax, ay), (bx, by) in [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]:
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        steps = max(1, length // (gap * 2))
        for i in range(steps):
            t0 = (2 * i * gap) / length
            t1 = min(1.0, (2 * i + 1) * gap / length)
            p0 = (int(ax + t0 * (bx - ax)), int(ay + t0 * (by - ay)))
            p1 = (int(ax + t1 * (bx - ax)), int(ay + t1 * (by - ay)))
            segs.append((p0, p1))
    return segs
