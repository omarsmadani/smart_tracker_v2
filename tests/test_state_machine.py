"""Unit tests for StateMachine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from tracker.state_machine import State, StateMachine

CFG = {
    "tau_reliable": 0.6,
    "tau_lost": 0.2,
    "tau_uncertain_write": 0.5,
    "k_confirm": 3,
    "n_lost_frames": 5,
    "t_max_lost": 150,
    "bbox_shrink_threshold": 0.4,
    "bbox_grow_threshold": 2.5,
    "bbox_check_window": 5,
}


def make_sm() -> StateMachine:
    return StateMachine(CFG)


# ── TRACKING transitions ──────────────────────────────────────────────

def test_stays_tracking_on_high_conf():
    sm = make_sm()
    for _ in range(10):
        assert sm.update(0.8) == State.TRACKING


def test_tracking_to_uncertain_after_k_confirm_low_conf():
    sm = make_sm()
    for _ in range(CFG["k_confirm"]):
        sm.update(0.4)          # below tau_reliable, above tau_lost
    assert sm.state == State.UNCERTAIN


def test_tracking_not_uncertain_before_k_confirm():
    sm = make_sm()
    for _ in range(CFG["k_confirm"] - 1):
        sm.update(0.4)
    assert sm.state == State.TRACKING


def test_tracking_to_lost_directly_on_n_lost_very_low_conf():
    sm = make_sm()
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.1)          # below tau_lost
    assert sm.state == State.LOST


# ── UNCERTAIN transitions ─────────────────────────────────────────────

def test_uncertain_returns_to_tracking():
    sm = make_sm()
    # Drive into UNCERTAIN
    for _ in range(CFG["k_confirm"]):
        sm.update(0.4)
    assert sm.state == State.UNCERTAIN
    # High-conf frames → back to TRACKING
    for _ in range(CFG["k_confirm"]):
        sm.update(0.9)
    assert sm.state == State.TRACKING


def test_uncertain_to_lost():
    sm = make_sm()
    for _ in range(CFG["k_confirm"]):
        sm.update(0.4)
    assert sm.state == State.UNCERTAIN
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.0)
    assert sm.state == State.LOST


# ── LOST transitions ──────────────────────────────────────────────────

def test_lost_recovers_on_high_conf():
    sm = make_sm()
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.0)
    assert sm.state == State.LOST
    sm.update(0.9)
    assert sm.state == State.RECOVERED


def test_force_recovered():
    sm = make_sm()
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.0)
    sm.force_recovered()
    assert sm.state == State.RECOVERED


def test_lost_stays_lost_under_t_max():
    sm = make_sm()
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.0)
    for _ in range(50):
        sm.update(0.1)          # low but above tau_lost — stays LOST
    assert sm.state == State.LOST


# ── RECOVERED transitions ─────────────────────────────────────────────

def test_recovered_to_tracking_after_k_confirm():
    sm = make_sm()
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.0)
    sm.update(0.9)              # → RECOVERED
    for _ in range(CFG["k_confirm"]):
        sm.update(0.9)
    assert sm.state == State.TRACKING


def test_recovered_to_lost_on_low_conf():
    sm = make_sm()
    for _ in range(CFG["n_lost_frames"]):
        sm.update(0.0)
    sm.update(0.9)              # → RECOVERED
    sm.update(0.0)              # below tau_lost → LOST
    assert sm.state == State.LOST


# ── Bbox stability ────────────────────────────────────────────────────

def test_bbox_shrink_forces_uncertain():
    sm = make_sm()
    # Fill window with a reference area
    for _ in range(CFG["bbox_check_window"]):
        sm.update(0.9, bbox=(0, 0, 100, 100))
    assert sm.state == State.TRACKING
    # Dramatic shrink (10x10 vs 100x100 → ratio 0.01 < 0.4)
    sm.update(0.9, bbox=(0, 0, 10, 10))
    # One more frame to exceed k_confirm streak
    for _ in range(CFG["k_confirm"] - 1):
        sm.update(0.9, bbox=(0, 0, 10, 10))
    assert sm.state == State.UNCERTAIN


def test_bbox_grow_forces_uncertain():
    sm = make_sm()
    for _ in range(CFG["bbox_check_window"]):
        sm.update(0.9, bbox=(0, 0, 10, 10))
    assert sm.state == State.TRACKING
    # Dramatic grow (500x500 vs 10x10 → ratio 2500 > 2.5)
    for _ in range(CFG["k_confirm"]):
        sm.update(0.9, bbox=(0, 0, 500, 500))
    assert sm.state == State.UNCERTAIN
