"""Tests for gaze estimation module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gaze import GazeEstimator
from src.schema import GazeSource, FocusZone, KeypointRecord


def _make_kpt(face_conf: float = 0.9, nose_xy=(960, 400)):
    """Create a mock KeypointRecord with controlled face confidence."""
    pts = []
    # 0: nose
    pts.append([float(nose_xy[0]), float(nose_xy[1]), face_conf])
    # 1: left_eye
    pts.append([nose_xy[0] - 30, nose_xy[1] - 20, face_conf])
    # 2: right_eye
    pts.append([nose_xy[0] + 30, nose_xy[1] - 20, face_conf])
    # 3: left_ear
    pts.append([nose_xy[0] - 60, nose_xy[1] - 5, face_conf])
    # 4: right_ear
    pts.append([nose_xy[0] + 60, nose_xy[1] - 5, face_conf])
    # 5: left_shoulder
    pts.append([nose_xy[0] - 100, nose_xy[1] + 150, 0.9])
    # 6: right_shoulder
    pts.append([nose_xy[0] + 100, nose_xy[1] + 150, 0.9])
    # 7-16: remaining body points (rough positions)
    for i in range(10):
        pts.append([nose_xy[0] + (i - 5) * 20, nose_xy[1] + 200 + i * 30, 0.7])

    return KeypointRecord(points=pts, mean_confidence=face_conf)


class TestGazeEstimator:
    def test_pnp_path(self):
        """With high face confidence, PnP path should be used."""
        estimator = GazeEstimator(min_face_conf=0.4)
        kpt = _make_kpt(face_conf=0.9)
        result = estimator.estimate(kpt, frame_shape=(1080, 1920))
        assert result.source == GazeSource.PNP
        assert 0.0 <= result.focus_score <= 1.0

    def test_fallback_path(self):
        """With low face confidence, fallback should be used."""
        estimator = GazeEstimator(min_face_conf=0.9)
        kpt = _make_kpt(face_conf=0.2)
        result = estimator.estimate(kpt, frame_shape=(1080, 1920))
        assert result.source in (GazeSource.FALLBACK, GazeSource.PRIOR)

    def test_prior_path(self):
        """With all keypoints zero confidence, prior should be returned."""
        estimator = GazeEstimator(min_face_conf=0.5)
        pts = [[0, 0, 0.0]] * 17
        kpt = KeypointRecord(points=pts, mean_confidence=0.0)
        result = estimator.estimate(kpt, frame_shape=(1080, 1920))
        assert result.source in (GazeSource.FALLBACK, GazeSource.PRIOR)

    def test_focus_score_range(self):
        estimator = GazeEstimator()
        kpt = _make_kpt(face_conf=0.95)
        result = estimator.estimate(kpt)
        assert 0.0 <= result.focus_score <= 1.0
