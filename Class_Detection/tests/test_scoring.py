"""Tests for CAS / CTES scoring engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring import calc_cas, calc_ctes, compute_classroom_metrics
from src.schema import StudentState, ActionRecord, GazeRecord


class TestCalcCAS:
    def test_weighted_fusion(self):
        """CAS should take the weighted average of action and gaze."""
        # Default weights: w_action=0.6, w_gaze=0.4
        # 0.9 * 0.6 + 0.3 * 0.4 = 0.54 + 0.12 = 0.66
        assert abs(calc_cas(0.9, 0.3) - 0.66) < 1e-6
        # 0.3 * 0.6 + 0.9 * 0.4 = 0.18 + 0.36 = 0.54
        assert abs(calc_cas(0.3, 0.9) - 0.54) < 1e-6

    def test_clamped_to_01(self):
        assert calc_cas(1.5, 0.5) == 1.0
        assert calc_cas(-0.1, 0.0) == 0.0

    def test_weights(self):
        # (0.5 * 2.0 + 0.8 * 1.0) / 3.0 = 1.8 / 3.0 = 0.6
        assert abs(calc_cas(0.5, 0.8, w_action=2.0, w_gaze=1.0) - 0.6) < 1e-6
        # (0.5 * 0.5 + 0.8 * 1.0) / 1.5 = (0.25 + 0.8) / 1.5 = 1.05 / 1.5 = 0.7
        result = calc_cas(0.5, 0.8, w_action=0.5, w_gaze=1.0)
        assert abs(result - 0.7) < 1e-6

    def test_negative_penalty(self):
        """Negative behaviors should trigger a 0.5x penalty."""
        # Normal: 0.8*0.6 + 0.8*0.4 = 0.48 + 0.32 = 0.8
        # With penalty: 0.8 * 0.5 = 0.4
        assert abs(calc_cas(0.8, 0.8, action_label="using_phone") - 0.4) < 1e-6
        assert abs(calc_cas(0.8, 0.8, action_label="sleeping") - 0.4) < 1e-6
        assert abs(calc_cas(0.8, 0.8, action_label="yawning") - 0.4) < 1e-6



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
