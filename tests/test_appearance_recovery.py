"""Unit tests for AppearanceRecovery. No models — synthetic L2-normalised vectors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from tracker.appearance_recovery import AppearanceRecovery

CFG = {"tau_reid": 0.6, "max_candidates": 20}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_ar() -> AppearanceRecovery:
    return AppearanceRecovery(CFG)


# ── basic matching ────────────────────────────────────────────────────

def test_exact_match_returns_correct_index():
    ar = make_ar()
    fp = unit(np.array([1.0, 0.0, 0.0]))
    ar.store_fingerprint([fp])

    # identical vector → cosine = 1.0
    candidates = [unit(np.array([0.0, 1.0, 0.0])),   # dissimilar
                  unit(np.array([1.0, 0.0, 0.0]))]    # identical
    assert ar.search(candidates) == 1


def test_no_match_below_threshold():
    ar = make_ar()
    fp = unit(np.array([1.0, 0.0, 0.0]))
    ar.store_fingerprint([fp])

    # orthogonal → cosine = 0.0
    candidates = [unit(np.array([0.0, 1.0, 0.0]))]
    assert ar.search(candidates) is None


def test_returns_none_with_empty_fingerprint():
    ar = make_ar()
    candidates = [unit(np.array([1.0, 0.0, 0.0]))]
    assert ar.search(candidates) is None


def test_returns_none_with_empty_candidates():
    ar = make_ar()
    ar.store_fingerprint([unit(np.array([1.0, 0.0, 0.0]))])
    assert ar.search([]) is None


# ── threshold boundary ────────────────────────────────────────────────

def test_exactly_at_threshold_not_matched():
    ar = AppearanceRecovery({"tau_reid": 0.9, "max_candidates": 20})
    fp = unit(np.array([1.0, 0.0]))
    ar.store_fingerprint([fp])
    # cosine exactly 0.9 is NOT above threshold (strictly greater required)
    angle = np.arccos(0.9)
    cand = unit(np.array([np.cos(angle), np.sin(angle)]))
    assert abs(float(np.dot(cand, fp)) - 0.9) < 1e-6
    assert ar.search([cand]) is None


def test_just_above_threshold_matched():
    ar = AppearanceRecovery({"tau_reid": 0.6, "max_candidates": 20})
    fp = unit(np.array([1.0, 0.0]))
    ar.store_fingerprint([fp])
    angle = np.arccos(0.65)
    cand = unit(np.array([np.cos(angle), np.sin(angle)]))
    assert ar.search([cand]) == 0


# ── multi-fingerprint mean ────────────────────────────────────────────

def test_mean_over_multiple_fingerprints():
    ar = make_ar()
    # Two fingerprints: one close, one far — mean should still clear threshold
    fp_close = unit(np.array([1.0, 0.01, 0.0]))
    fp_far   = unit(np.array([1.0, 0.0,  0.0]))
    ar.store_fingerprint([fp_close, fp_far])
    cand = unit(np.array([1.0, 0.0, 0.0]))
    assert ar.search([cand]) == 0


# ── max_candidates cap ────────────────────────────────────────────────

def test_candidates_capped_at_max():
    cfg = {"tau_reid": 0.6, "max_candidates": 2}
    ar = AppearanceRecovery(cfg)
    fp = unit(np.array([1.0, 0.0]))
    ar.store_fingerprint([fp])

    # Perfect match is index 3 — beyond the cap of 2 → should not be found
    candidates = [
        unit(np.array([0.0, 1.0])),   # idx 0: orthogonal
        unit(np.array([0.0, 1.0])),   # idx 1: orthogonal
        unit(np.array([1.0, 0.0])),   # idx 2: perfect — but beyond cap
    ]
    assert ar.search(candidates) is None


# ── fingerprint replacement ───────────────────────────────────────────

def test_store_fingerprint_replaces_previous():
    ar = make_ar()
    old = unit(np.array([1.0, 0.0]))
    ar.store_fingerprint([old])

    new = unit(np.array([0.0, 1.0]))
    ar.store_fingerprint([new])

    # old fingerprint gone — candidate matching old should fail
    assert ar.search([old]) is None
    # candidate matching new should succeed
    assert ar.search([new]) == 0
