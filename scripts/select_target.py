"""First-frame target selection: show detections, user clicks one, return bbox + class."""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracker.detector import Detection, Detector

WINDOW = "Select target (click bbox, press q to cancel)"


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


def select(frame: np.ndarray, detector: Detector,
           labels_path: Optional[Path] = None) -> Optional[Tuple[Tuple[int,int,int,int], int, str]]:
    detections = detector.detect(frame)
    if not detections:
        print("No detections in first frame.")
        return None

    labels = _load_labels(labels_path) if labels_path else []
    canvas = frame.copy()
    for i, d in enumerate(detections):
        x, y, w, h = d.bbox
        name = labels[d.class_id] if d.class_id < len(labels) else str(d.class_id)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, f"{i}:{name} {d.confidence:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    selected: dict = {"det": None}

    def on_mouse(event, px, py, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Prefer smallest bbox containing the click (more specific)
        hits = [d for d in detections if _inside(px, py, d.bbox)]
        if not hits:
            return
        hits.sort(key=lambda d: d.bbox[2] * d.bbox[3])
        selected["det"] = hits[0]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if selected["det"] is not None:
            break
        if key == ord("q"):
            cv2.destroyWindow(WINDOW)
            return None
    cv2.destroyWindow(WINDOW)

    d: Detection = selected["det"]
    name = labels[d.class_id] if d.class_id < len(labels) else str(d.class_id)
    return (d.bbox, d.class_id, name)
