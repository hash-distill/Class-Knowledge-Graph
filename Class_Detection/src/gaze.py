"""Head-pose estimation via PnP with torso-vector fallback.

Converts YOLO26-Pose face keypoints into Euler angles (pitch, yaw, roll)
and maps them to a focus-score in [0, 1].

Pipeline priority:
  1. PnP from 5 face keypoints (nose, eyes, ears)  → accurate Euler angles
  2. Torso-vector fallback (shoulder midpoint → nose) → coarse direction
  3. Neutral prior (0.55)                             → no keypoint data
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.schema import FocusZone, GazeRecord, GazeSource, KeypointRecord


# Keypoint indices (duplicated from pose.py to avoid ultralytics dependency)
_FACE_INDICES = [0, 1, 2, 3, 4]          # nose, left_eye, right_eye, left_ear, right_ear
_SHOULDER_INDICES = [5, 6]                # left_shoulder, right_shoulder


def _get_face_points(kpt: KeypointRecord) -> np.ndarray:
    return np.array(kpt.points)[_FACE_INDICES]


def _get_shoulder_points(kpt: KeypointRecord) -> np.ndarray:
    return np.array(kpt.points)[_SHOULDER_INDICES]


def _face_confidence(kpt: KeypointRecord) -> float:
    return float(np.mean(_get_face_points(kpt)[:, 2]))


# Standard 3D face model (approximate, millimeters)
# 6 points required by OpenCV DLT-based solvePnP
_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],            # nose tip        (kpt 0)
        [-30.0, 40.0, -30.0],       # left eye        (kpt 1)
        [30.0, 40.0, -30.0],        # right eye       (kpt 2)
        [-60.0, -10.0, -50.0],      # left ear        (kpt 3)
        [60.0, -10.0, -50.0],       # right ear       (kpt 4)
        [0.0, -70.0, -10.0],        # chin midpoint   (estimated from kpts)
    ],
    dtype=np.float64,
)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _in_range(val: float, rng: list[float]) -> bool:
    return rng[0] <= val <= rng[1]


class GazeEstimator:
    """Estimate gaze direction and focus score for a single person."""

    def __init__(
        self,
        min_face_conf: float = 0.4,
        focus_zones: Optional[dict] = None,
        fallback_base: float = 0.55,
    ) -> None:
        self.min_face_conf = min_face_conf
        self.fallback_base = fallback_base

        # Default focus-zone config
        self.zones = focus_zones or {
            "board_focus": {"pitch_range": [-15, 15], "yaw_range": [-20, 20], "score": 0.90},
            "desk_focus": {"pitch_range": [-50, -15], "yaw_range": [-15, 15], "score": 0.65},
        }
        self.wander_score = 0.20

    # ── public API ────────────────────────────────────────────

    def estimate(
        self,
        kpt: KeypointRecord,
        frame_shape: tuple[int, int] = (1080, 1920),
    ) -> GazeRecord:
        """Return a GazeRecord for a single person's keypoints.

        Parameters
        ----------
        kpt : KeypointRecord
            17-keypoint record from PoseEstimator.
        frame_shape : (height, width)
            Used to construct an approximate camera matrix.
        """
        face_conf = _face_confidence(kpt)

        if face_conf >= self.min_face_conf:
            result = self._pnp_estimate(kpt, frame_shape)
            if result is not None:
                return result

        # Fallback: torso direction
        result = self._torso_fallback(kpt)
        if result is not None:
            return result

        # Neutral prior
        return GazeRecord(
            focus_score=self.fallback_base,
            focus_zone=FocusZone.WANDERING,
            source=GazeSource.PRIOR,
        )

    # ── PnP path ─────────────────────────────────────────────

    def _pnp_estimate(
        self,
        kpt: KeypointRecord,
        frame_shape: tuple[int, int],
    ) -> Optional[GazeRecord]:
        face = _get_face_points(kpt)  # (5, 3) [x, y, conf]
        pts_5 = face[:, :2].astype(np.float64)

        # Estimate chin as: nose + 1.5 * (nose - eye_midpoint)
        eye_mid = (pts_5[1] + pts_5[2]) / 2.0
        chin_est = pts_5[0] + 1.5 * (pts_5[0] - eye_mid)
        pts_2d = np.vstack([pts_5, chin_est.reshape(1, 2)])  # (6, 2)

        h, w = frame_shape
        focal = float(w)  # rough focal length approximation
        cam_matrix = np.array(
            [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                _FACE_3D, pts_2d, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            return None
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_to_euler(rmat)

        zone, score = self._classify_gaze(pitch, yaw)
        return GazeRecord(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            focus_score=round(score, 4),
            focus_zone=zone,
            source=GazeSource.PNP,
        )

    # ── torso fallback ───────────────────────────────────────

    def _torso_fallback(self, kpt: KeypointRecord) -> Optional[GazeRecord]:
        shoulders = _get_shoulder_points(kpt)
        nose = np.array(kpt.points[0])

        if shoulders[0, 2] < 0.3 or shoulders[1, 2] < 0.3 or nose[2] < 0.3:
            return None

        mid_sh = (shoulders[0, :2] + shoulders[1, :2]) / 2.0
        direction = nose[:2] - mid_sh
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None

        # Positive y (downward in image) + small x → roughly facing forward
        cos_angle = float(-direction[1] / norm)  # negative because y grows down
        score = _clamp(0.35 + 0.55 * ((cos_angle + 1.0) / 2.0))

        return GazeRecord(
            focus_score=round(score, 4),
            focus_zone=FocusZone.BOARD if score > 0.6 else FocusZone.WANDERING,
            source=GazeSource.FALLBACK,
        )

    # ── helpers ──────────────────────────────────────────────

    def _classify_gaze(self, pitch: float, yaw: float) -> tuple[FocusZone, float]:
        for zone_name, cfg in self.zones.items():
            if _in_range(pitch, cfg["pitch_range"]) and _in_range(yaw, cfg["yaw_range"]):
                return FocusZone(zone_name), float(cfg["score"])
        return FocusZone.WANDERING, self.wander_score

    @staticmethod
    def _rotation_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
        """Convert 3x3 rotation matrix to Euler angles in degrees."""
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0
        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
