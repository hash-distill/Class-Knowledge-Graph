"""Tests for CAS / CTES scoring engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring import calc_cas, calc_ctes, compute_classroom_metrics
from src.schema import StudentState, ActionRecord, GazeRecord


class TestCalcCAS:
    def test_max_fusion(self):
        """CAS should take the max of weighted action and gaze."""
        # max(0.6*0.9, 0.4*0.3) = max(0.54, 0.12) = 0.54
        assert abs(calc_cas(0.9, 0.3) - 0.54) < 1e-6
        # max(0.6*0.3, 0.4*0.9) = max(0.18, 0.36) = 0.36
        assert abs(calc_cas(0.3, 0.9) - 0.36) < 1e-6

    def test_high_action_low_gaze(self):
        """Student writing (high action) but looking down (low gaze) → still engaged."""
        cas = calc_cas(0.85, 0.20)
        # max(0.6*0.85, 0.4*0.20) = max(0.51, 0.08) = 0.51
        assert abs(cas - 0.51) < 1e-6

    def test_clamped_to_01(self):
        # max(0.6*2.0, 0.4*0.5) = max(1.2, 0.2) = 1.2 → clamped to 1.0
        assert calc_cas(2.0, 0.5) == 1.0
        assert calc_cas(-0.1, 0.0) == 0.0

    def test_weights(self):
        # max(2.0*0.5, 1.0*0.8) = max(1.0, 0.8) = 1.0 (clamped)
        assert abs(calc_cas(0.5, 0.8, w_action=2.0, w_gaze=1.0) - 1.0) < 1e-6
        # max(0.5*0.5, 1.0*0.8) = max(0.25, 0.8) = 0.8
        result = calc_cas(0.5, 0.8, w_action=0.5, w_gaze=1.0)
        assert abs(result - 0.8) < 1e-6

    def test_negative_penalty(self):
        """Negative behaviors should trigger a 0.5x penalty."""
        # max(0.6*0.8, 0.4*0.8) = max(0.48, 0.32) = 0.48
        # With penalty: 0.48 * 0.5 = 0.24
        assert abs(calc_cas(0.8, 0.8, action_label="using_phone") - 0.24) < 1e-6
        assert abs(calc_cas(0.8, 0.8, action_label="sleeping") - 0.24) < 1e-6
        assert abs(calc_cas(0.8, 0.8, action_label="yawning") - 0.24) < 1e-6



class TestCalcCTES:
    def test_uniform_class(self):
        """All students same CAS → zero variance → CTES = mean."""
        vals = [0.8] * 10
        ctes = calc_ctes(vals)
        assert abs(ctes - 0.8) < 1e-6

    def test_polarized_class(self):
        """High variance should penalize CTES."""
        vals = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ctes = calc_ctes(vals)
        assert ctes < 0.5, f"Polarized CTES should be low, got {ctes}"

    def test_empty(self):
        assert calc_ctes([]) == 0.0

    def test_single_student(self):
        ctes = calc_ctes([0.7])
        assert abs(ctes - 0.7) < 1e-6


class TestComputeClassroomMetrics:
    def test_basic(self):
        students = [
            StudentState(
                track_id=1, bbox=[0, 0, 100, 100],
                action=ActionRecord(label="write", confidence=0.8),
                gaze=GazeRecord(focus_score=0.7),
                cas=0.8,
            ),
            StudentState(
                track_id=2, bbox=[0, 0, 100, 100],
                action=ActionRecord(label="read", confidence=0.7),
                gaze=GazeRecord(focus_score=0.6),
                cas=0.7,
            ),
        ]
        metrics = compute_classroom_metrics(students)
        assert metrics.active_tracks == 2
        assert metrics.mean_cas == 0.75
        assert metrics.ctes_score > 0
        assert "write" in metrics.behavior_distribution.counts

    def test_empty(self):
        metrics = compute_classroom_metrics([])
        assert metrics.active_tracks == 0
        assert metrics.ctes_score == 0.0
