"""Multi-scale template search with NMS for recovery candidate generation."""
from typing import List, Tuple

import cv2
import numpy as np

Bbox = Tuple[int, int, int, int]  # (x, y, w, h)


def search(
    frame: np.ndarray,
    seed_crop: np.ndarray,
    scales: List[float],
    top_k: int,
    method: int = cv2.TM_CCOEFF_NORMED,
    nms_distance_ratio: float = 0.5,
) -> List[Tuple[Bbox, float]]:
    """Find seed_crop in frame at multiple scales using matchTemplate.

    Returns up to top_k (bbox, score) tuples sorted by score descending.
    NMS suppresses peaks within nms_distance_ratio * template_size of a stronger peak.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    seed_gray = cv2.cvtColor(seed_crop, cv2.COLOR_BGR2GRAY) if seed_crop.ndim == 3 else seed_crop

    candidates: List[Tuple[float, int, int, int, int]] = []  # (score, x, y, w, h)

    for scale in scales:
        th = max(1, int(round(seed_gray.shape[0] * scale)))
        tw = max(1, int(round(seed_gray.shape[1] * scale)))

        if th > frame_gray.shape[0] or tw > frame_gray.shape[1]:
            continue

        tmpl = cv2.resize(seed_gray, (tw, th))
        result = cv2.matchTemplate(frame_gray, tmpl, method)

        # Collect local maxima via repeated minMaxLoc with masking
        result_copy = result.copy()
        suppress_radius_y = max(1, int(th * nms_distance_ratio))
        suppress_radius_x = max(1, int(tw * nms_distance_ratio))

        for _ in range(top_k):
            _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
            if max_val <= 0:
                break
            mx, my = max_loc
            candidates.append((float(max_val), mx, my, tw, th))
            # suppress neighbourhood
            y0 = max(0, my - suppress_radius_y)
            y1 = min(result_copy.shape[0], my + suppress_radius_y + 1)
            x0 = max(0, mx - suppress_radius_x)
            x1 = min(result_copy.shape[1], mx + suppress_radius_x + 1)
            result_copy[y0:y1, x0:x1] = 0.0

    # Global NMS across scales: sort by score, suppress duplicates
    candidates.sort(key=lambda c: c[0], reverse=True)
    kept: List[Tuple[Bbox, float]] = []
    suppressed = [False] * len(candidates)

    for i, (score_i, xi, yi, wi, hi) in enumerate(candidates):
        if suppressed[i]:
            continue
        kept.append(((xi, yi, wi, hi), score_i))
        if len(kept) >= top_k:
            break
        cx_i = xi + wi / 2
        cy_i = yi + hi / 2
        for j in range(i + 1, len(candidates)):
            if suppressed[j]:
                continue
            _, xj, yj, wj, hj = candidates[j]
            cx_j = xj + wj / 2
            cy_j = yj + hj / 2
            avg_w = (wi + wj) / 2
            avg_h = (hi + hj) / 2
            if (abs(cx_i - cx_j) < avg_w * nms_distance_ratio and
                    abs(cy_i - cy_j) < avg_h * nms_distance_ratio):
                suppressed[j] = True

    return kept
