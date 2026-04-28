"""Unit tests for multi-scale template search + NMS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pytest
from tracker.template_search import search


def _blank_frame(h=300, w=400):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paste(frame, crop, x, y):
    """Paste crop into frame at (x, y) top-left."""
    h, w = crop.shape[:2]
    frame[y: y + h, x: x + w] = crop
    return frame


def _make_seed(size=40, color=(200, 100, 50)):
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:] = color
    cv2.rectangle(crop, (5, 5), (size - 5, size - 5), (255, 255, 255), 2)
    return crop


# ── basic hit ────────────────────────────────────────────────────────

def test_finds_exact_match():
    seed = _make_seed(40)
    frame = _blank_frame()
    _paste(frame, seed, 80, 60)
    results = search(frame, seed, scales=[1.0], top_k=1)
    assert len(results) == 1
    bbox, score = results[0]
    assert score > 0.9
    x, y, w, h = bbox
    assert abs(x - 80) <= 2
    assert abs(y - 60) <= 2


def test_finds_multiple_locations():
    seed = _make_seed(30)
    frame = _blank_frame(300, 500)
    _paste(frame, seed, 50, 50)
    _paste(frame, seed, 200, 150)
    results = search(frame, seed, scales=[1.0], top_k=5)
    assert len(results) >= 2
    xs = [r[0][0] for r in results]
    assert any(abs(x - 50) <= 3 for x in xs)
    assert any(abs(x - 200) <= 3 for x in xs)


def test_multi_scale_finds_scaled_copy():
    seed = _make_seed(40)
    small_seed = cv2.resize(seed, (28, 28))  # ~ 0.7x scale
    frame = _blank_frame(300, 400)
    _paste(frame, small_seed, 100, 80)
    results = search(frame, seed, scales=[0.7, 1.0, 1.4], top_k=3)
    assert len(results) >= 1
    best_score = results[0][1]
    assert best_score > 0.5


def test_no_match_gives_low_score():
    seed = _make_seed(40, color=(200, 100, 50))
    # Frame is completely different color
    frame = np.full((300, 400, 3), fill_value=30, dtype=np.uint8)
    results = search(frame, seed, scales=[1.0], top_k=1)
    if results:
        best_score = results[0][1]
        # Score must be lower than a true match (> 0.9)
        assert best_score < 0.7


def test_nms_suppresses_duplicates():
    seed = _make_seed(40)
    frame = _blank_frame()
    _paste(frame, seed, 80, 60)
    # With tight nms_distance_ratio, two nearby peaks become one
    results_tight = search(frame, seed, scales=[1.0], top_k=5, nms_distance_ratio=0.5)
    # The same seed location should only produce one kept result near (80,60)
    close = [r for r in results_tight if abs(r[0][0] - 80) <= 5 and abs(r[0][1] - 60) <= 5]
    assert len(close) == 1


def test_returns_empty_when_seed_larger_than_frame():
    seed = _make_seed(200)
    frame = _blank_frame(50, 50)
    results = search(frame, seed, scales=[1.0], top_k=3)
    assert results == []


def test_top_k_respected():
    seed = _make_seed(20)
    frame = _blank_frame(300, 500)
    # Paste seed at 4 distinct locations
    for x, y in [(30, 30), (150, 30), (30, 150), (150, 150)]:
        _paste(frame, seed, x, y)
    results = search(frame, seed, scales=[1.0], top_k=2)
    assert len(results) <= 2


def test_results_sorted_descending():
    seed = _make_seed(30)
    frame = _blank_frame(300, 500)
    _paste(frame, seed, 50, 50)
    _paste(frame, seed, 250, 150)
    results = search(frame, seed, scales=[1.0], top_k=5)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
