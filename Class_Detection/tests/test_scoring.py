"""Tests for CAS / CTES scoring engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring import calc_cas, calc_ctes, compute_classroom_metrics
from src.schema import StudentState, ActionRecord, GazeRecord


class TestCalcCAS:
    def test_max_fusion(self):
        """CAS should take the max of weighted action and gaze."""
        assert calc_cas(0.9, 0.3) == 0.9
        assert calc_cas(0.3, 0.9) == 0.9

    def test_clamped_to_01(self):
        assert calc_cas(1.5, 0.5) == 1.0
        assert calc_cas(-0.1, 0.0) == 0.0

    def test_weights(self):
        assert calc_cas(0.5, 0.8, w_action=2.0, w_gaze=1.0) == 1.0
        result = calc_cas(0.5, 0.8, w_action=0.5, w_gaze=1.0)
        assert abs(result - 0.8) < 1e-6


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
