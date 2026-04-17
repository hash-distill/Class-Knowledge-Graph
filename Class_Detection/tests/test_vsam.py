"""Tests for VSAM Gaussian alignment."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vsam import gaussian_weight, score_knowledge_point, VSAMAligner


class TestGaussianWeight:
    def test_peak_at_center(self):
        """Weight should be 1.0 exactly at (t_ocr + mu)."""
        w = gaussian_weight(t=8.0, t_ocr=5.0, mu=3.0, sigma=1.5)
        assert abs(w - 1.0) < 1e-9

    def test_symmetric_decay(self):
        """Weights should decay symmetrically around center."""
        w_before = gaussian_weight(t=7.0, t_ocr=5.0, mu=3.0, sigma=1.5)
        w_after = gaussian_weight(t=9.0, t_ocr=5.0, mu=3.0, sigma=1.5)
        assert abs(w_before - w_after) < 1e-9

    def test_far_from_center(self):
        """Weight should be near zero far from the center."""
        w = gaussian_weight(t=100.0, t_ocr=5.0, mu=3.0, sigma=1.5)
        assert w < 1e-6

    def test_narrow_sigma(self):
        """Narrower sigma → sharper peak."""
        w_wide = gaussian_weight(t=7.0, t_ocr=5.0, mu=3.0, sigma=3.0)
        w_narrow = gaussian_weight(t=7.0, t_ocr=5.0, mu=3.0, sigma=0.5)
        assert w_narrow < w_wide


class TestScoreKnowledgePoint:
    def test_perfect_attention(self):
        """All CAS=1.0 → score should be 1.0 regardless of alignment."""
        cas = [1.0] * 10
        ts = [float(i) for i in range(10)]
        score = score_knowledge_point(cas, ts, t_ocr=0.0)
        assert abs(score - 1.0) < 1e-6

    def test_zero_attention(self):
        cas = [0.0] * 10
        ts = [float(i) for i in range(10)]
        score = score_knowledge_point(cas, ts, t_ocr=0.0)
        assert abs(score) < 1e-6

    def test_aligned_burst(self):
        """High CAS near (t_ocr+mu) should produce higher score than
        high CAS far from the center."""
        # Attention peak aligned with the Gaussian center
        cas_aligned = [0.2] * 5 + [0.9, 0.95, 1.0, 0.95, 0.9] + [0.2] * 5
        ts = [float(i) for i in range(15)]
        # t_ocr = 2, mu = 3 → center = 5, so peak at indices 5-9 is near center
        score_aligned = score_knowledge_point(cas_aligned, ts, t_ocr=2.0)

        # Attention peak misaligned
        cas_misaligned = [0.9, 0.95, 1.0, 0.95, 0.9] + [0.2] * 10
        score_misaligned = score_knowledge_point(cas_misaligned, ts, t_ocr=2.0)

        assert score_aligned > score_misaligned

    def test_empty_input(self):
        assert score_knowledge_point([], [], t_ocr=0.0) == 0.0

    def test_mismatched_lengths(self):
        assert score_knowledge_point([0.5, 0.6], [1.0], t_ocr=0.0) == 0.0


class TestVSAMAligner:
    def test_lifecycle(self):
        """trigger → feed → evaluate lifecycle."""
        aligner = VSAMAligner(mu=3.0, sigma=1.5, window_duration=10.0)

        aligner.trigger("二元一次方程", t_ocr=0.0)
        assert len(aligner.active_anchors) == 1

        for t in range(12):
            aligner.feed(float(t), 0.8)

        closed = aligner.evaluate(current_time=11.0)
        assert len(closed) == 1
        assert closed[0].entity == "二元一次方程"
        assert closed[0].score_k > 0
        assert len(aligner.active_anchors) == 0

    def test_multiple_anchors(self):
        aligner = VSAMAligner(mu=2.0, sigma=1.0, window_duration=5.0)
        aligner.trigger("知识点A", t_ocr=0.0)
        aligner.trigger("知识点B", t_ocr=3.0)

        for t in range(10):
            aligner.feed(float(t), 0.7)

        closed = aligner.evaluate(current_time=10.0)
        assert len(closed) == 2
