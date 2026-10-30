"""YOLO26-Pose keypoint extraction.

Extracts 17 COCO keypoints per detected person.  Used downstream by:
  - ``gaze.py``   → PnP head-pose from face keypoints [0..4]
  - ``action.py`` → ST-GCN from full 17-keypoint temporal window
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.schema import KeypointRecord


class PoseEstimator:
    """Thin wrapper around YOLO26 pose model."""

    # COCO 17 keypoint names for reference
    KPT_NAMES: list[str] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    FACE_INDICES = [0, 1, 2, 3, 4]          # nose, eyes, ears
    SHOULDER_INDICES = [5, 6]                # left / right shoulder

    def __init__(
        self,
        weights: str | Path = "yolo26n-pose.pt",
        device: str = "0",
        conf: float = 0.3,
        imgsz: int = 640,
    ) -> None:
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def estimate(self, frame: np.ndarray) -> tuple[list[KeypointRecord], list[list[float]]]:
        """Return keypoints and bboxes for every person detected in *frame*.

        Returns
        -------
        records : list[KeypointRecord]
            Keypoints for each detected person.
        bboxes : list[list[float]]
            Corresponding bounding boxes as [x1, y1, x2, y2].
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return [], []

        kpts = results[0].keypoints  # ultralytics Keypoints object
        boxes = results[0].boxes
        if kpts is None or kpts.data is None:
            return [], []

        records: list[KeypointRecord] = []
        bboxes: list[list[float]] = []
        data = kpts.data.cpu().numpy()  # (N, 17, 3)

        for i, person in enumerate(data):
            pts = person.tolist()  # list[list[float]]  (17, 3)
            mean_conf = float(np.mean(person[:, 2]))
            records.append(KeypointRecord(points=pts, mean_confidence=mean_conf))
            # Extract bbox from pose model's detection
            if boxes is not None and i < len(boxes):
                bbox = boxes[i].xyxy.squeeze(0).tolist()
                bboxes.append(bbox)
            else:
                bboxes.append([0.0, 0.0, 0.0, 0.0])

        return records, bboxes

    @staticmethod
    def get_face_points(kpt: KeypointRecord) -> np.ndarray:
        """Extract face keypoints as (5, 3) array [x, y, conf]."""
        pts = np.array(kpt.points)
        return pts[PoseEstimator.FACE_INDICES]

    @staticmethod
    def get_shoulder_points(kpt: KeypointRecord) -> np.ndarray:
        """Extract shoulder keypoints as (2, 3) array [x, y, conf]."""
        pts = np.array(kpt.points)
        return pts[PoseEstimator.SHOULDER_INDICES]

    @staticmethod
    def face_confidence(kpt: KeypointRecord) -> float:
        """Mean confidence of the five face keypoints."""
        face = PoseEstimator.get_face_points(kpt)
        return float(np.mean(face[:, 2]))
