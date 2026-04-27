"""Target selection UI: draws all detections, user clicks one to begin tracking."""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracker.detector import Detection, Detector

WINDOW = "Edge Tracker — Select Target"

_COL_DEFAULT  = (0, 220, 0)    # green  — detected, not hovered
_COL_HOVER    = (0, 220, 255)  # yellow — mouse is over this box
_COL_SELECTED = (0, 255, 255)  # cyan   — confirmed selection flash


def _load_labels(path: Path) -> List[str]:
    labels: List[str] = []
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        labels.append(parts[1] if len(parts) == 2 else line.strip())
    return labels


def _inside(px: int, py: int, bbox) -> bool:
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def _smallest_hit(detections: List[Detection], px: int, py: int) -> Optional[Detection]:
    hits = [d for d in detections if _inside(px, py, d.bbox)]
    if not hits:
        return None
    hits.sort(key=lambda d: d.bbox[2] * d.bbox[3])
    return hits[0]


def _draw_frame(base: np.ndarray, detections: List[Detection],
                labels: List[str], hover: Optional[Detection]) -> np.ndarray:
    canvas = base.copy()

    for d in detections:
        x, y, w, h = d.bbox
        name = labels[d.class_id] if d.class_id < len(labels) else f"class {d.class_id}"
        is_hover = (hover is not None and d is hover)
        color = _COL_HOVER if is_hover else _COL_DEFAULT
        thickness = 3 if is_hover else 2

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        label = f"{name}  {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(th + 4, y - 4)
        cv2.rectangle(canvas, (x, ly - th - 4), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(canvas, label, (x + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Instruction banner at the bottom
    h_img = canvas.shape[0]
    msg = "Click an object to track    |    Q = cancel"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(canvas, (0, h_img - th - 16), (canvas.shape[1], h_img), (0, 0, 0), -1)
    cx = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, msg, (cx, h_img - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if not detections:
        nd = "No objects detected — waiting..."
        (tw, th), _ = cv2.getTextSize(nd, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cx = (canvas.shape[1] - tw) // 2
        cy = canvas.shape[0] // 2
        cv2.putText(canvas, nd, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return canvas


def select(
    frame: np.ndarray,
    detector: Detector,
    labels_path: Optional[Path] = None,
    cap: Optional[cv2.VideoCapture] = None,
) -> Optional[Tuple[Tuple[int, int, int, int], int, str]]:
    """Show selection UI. If `cap` is provided, reads live frames (camera mode).

    Returns (bbox, class_id, name) or None if cancelled.
    """
    labels = _load_labels(labels_path) if labels_path else []
    mouse: dict = {"x": 0, "y": 0, "clicked": None}

    def on_mouse(event, px, py, flags, param):
        mouse["x"] = px
        mouse["y"] = py
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["clicked"] = (px, py)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    current_frame = frame.copy()
    detections = detector.detect(current_frame)

    while True:
        # Live camera: refresh frame + detections continuously
        if cap is not None and cap.isOpened():
            ok, live = cap.read()
            if ok:
                current_frame = live
                detections = detector.detect(current_frame)

        hover = _smallest_hit(detections, mouse["x"], mouse["y"])
        canvas = _draw_frame(current_frame, detections, labels, hover)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyWindow(WINDOW)
            return None

        if mouse["clicked"] is not None:
            px, py = mouse["clicked"]
            mouse["clicked"] = None
            hit = _smallest_hit(detections, px, py)
            if hit is None:
                continue  # clicked empty space — keep waiting

            # Confirmation flash: draw selected box in cyan for 3 frames
            flash = current_frame.copy()
            x, y, w, h = hit.bbox
            cv2.rectangle(flash, (x, y), (x + w, y + h), _COL_SELECTED, 4)
            name = labels[hit.class_id] if hit.class_id < len(labels) else f"class {hit.class_id}"
            cv2.putText(flash, f"Tracking: {name}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COL_SELECTED, 2, cv2.LINE_AA)
            for _ in range(3):
                cv2.imshow(WINDOW, flash)
                cv2.waitKey(80)

            cv2.destroyWindow(WINDOW)
            return (hit.bbox, hit.class_id, name)
