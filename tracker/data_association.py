"""IoU-based data association between detections and the Kalman prediction."""
from typing import List, Optional, Tuple

from .detector import Detection

Bbox = Tuple[float, float, float, float]


def iou(a: Bbox, b: Bbox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detection(
    detections: List[Detection],
    predicted_bbox: Bbox,
    min_iou: float,
    target_class: Optional[int] = None,
) -> Optional[Detection]:
    """Return the detection with highest IoU vs prediction (above min_iou), else None."""
    best: Optional[Detection] = None
    best_iou = min_iou
    for d in detections:
        if target_class is not None and d.class_id != target_class:
            continue
        score = iou(d.bbox, predicted_bbox)
        if score > best_iou:
            best_iou = score
            best = d
    return best
