"""Unit tests for SmartMemory. No models needed — mock feature vectors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from tracker.smart_memory import SmartMemory
from tracker.state_machine import State

CFG = {
    "stm_size": 7,
    "ltm_size": 5,
    "ltm_min_confidence": 0.7,
}


def feat(val: float = 1.0) -> np.ndarray:
    v = np.array([val, 0.0], dtype=np.float32)
    return v / np.linalg.norm(v)


def make_mem() -> SmartMemory:
    return SmartMemory(CFG)


# ── STM ───────────────────────────────────────────────────────────────

def test_stm_fills_on_tracking():
    m = make_mem()
    for i in range(4):
        m.update(feat(), 0.8, State.TRACKING)
    assert m.stm_count == 4


def test_stm_capped_at_stm_size():
    m = make_mem()
    for _ in range(CFG["stm_size"] + 3):
        m.update(feat(), 0.8, State.TRACKING)
    assert m.stm_count == CFG["stm_size"]


def test_stm_no_write_during_lost():
    m = make_mem()
    for _ in range(3):
        m.update(feat(), 0.0, State.LOST)
    assert m.stm_count == 0


def test_stm_write_on_recovered():
    m = make_mem()
    m.update(feat(), 0.8, State.RECOVERED)
    assert m.stm_count == 1


# ── UNCERTAIN write threshold ─────────────────────────────────────────

def test_uncertain_write_above_threshold():
    m = make_mem()
    m.update_uncertain(feat(), 0.6, tau_uncertain_write=0.5)
    assert m.stm_count == 1


def test_uncertain_no_write_below_threshold():
    m = make_mem()
    m.update_uncertain(feat(), 0.3, tau_uncertain_write=0.5)
    assert m.stm_count == 0


# ── LTM ───────────────────────────────────────────────────────────────

def test_ltm_fills_on_high_conf_tracking():
    m = make_mem()
    for _ in range(CFG["ltm_size"]):
        m.update(feat(), 0.9, State.TRACKING)
    assert m.ltm_count == CFG["ltm_size"]


def test_ltm_ignores_low_conf():
    m = make_mem()
    for _ in range(5):
        m.update(feat(), 0.5, State.TRACKING)  # below ltm_min_confidence=0.7
    assert m.ltm_count == 0


def test_ltm_frozen_during_uncertain():
    m = make_mem()
    for _ in range(CFG["ltm_size"]):
        m.update(feat(), 0.9, State.TRACKING)
    assert m.ltm_count == CFG["ltm_size"]
    pre_features = m.get_ltm_features()[:]
    # Even if update_uncertain is called with high conf, LTM stays frozen
    m.update_uncertain(feat(0.5), 0.9, tau_uncertain_write=0.5)
    assert m.ltm_count == CFG["ltm_size"]
    for pre, post in zip(pre_features, m.get_ltm_features()):
        assert np.allclose(pre, post)


def test_ltm_capped_keeps_best():
    m = make_mem()
    # Fill LTM with medium entries
    for i in range(CFG["ltm_size"]):
        m.update(feat(float(i)), 0.75, State.TRACKING)
    assert m.ltm_count == CFG["ltm_size"]
    # Insert one with very high conf — should displace the lowest
    m.update(feat(99.0), 0.99, State.TRACKING)
    confs = [e.confidence for e in m._ltm]
    assert 0.99 in confs
    assert len(confs) == CFG["ltm_size"]


def test_ltm_frozen_during_lost():
    m = make_mem()
    for _ in range(CFG["ltm_size"]):
        m.update(feat(), 0.9, State.TRACKING)
    before = m.ltm_count
    m.update(feat(), 0.9, State.LOST)   # should be ignored
    assert m.ltm_count == before


# ── get_features ─────────────────────────────────────────────────────

def test_get_features_returns_stm_and_ltm():
    m = make_mem()
    for _ in range(3):
        m.update(feat(), 0.9, State.TRACKING)
    all_f = m.get_features()
    assert len(all_f) == m.stm_count + m.ltm_count
