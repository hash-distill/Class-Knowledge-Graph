"""YOLO26 detection and ByteTrack multi-object tracking.

This module wraps *ultralytics* YOLO26 for:
  1. Per-frame object detection (3 robust classes: student/teacher/screen_board).
  2. Video-level multi-object tracking with built-in ByteTrack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
from ultralytics import YOLO

from src.schema import BBoxRecord


class Detector:
    """YOLO26 detector with optional built-in tracking."""

    def __init__(
        self,
        weights: str | Path = "yolo26s.pt",
        device: str = "0",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 960,
    ) -> None:
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    # ── single-frame detection (no tracking) ──────────────────

    def detect_frame(self, frame: np.ndarray) -> list[BBoxRecord]:
        """Run detection on a single BGR frame and return records."""
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    # ── video tracking (ByteTrack) ────────────────────────────

    def track_video(
        self,
        source: str | Path,
        tracker: str = "bytetrack.yaml",
    ) -> Generator[list[BBoxRecord], None, None]:
        """Yield per-frame detection records with persistent track IDs.

        Uses the *ultralytics* built-in tracker so there is no need to
        install a separate ByteTrack package.
        """
        results_gen = self.model.track(
            source=str(source),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=tracker,
            persist=True,
            stream=True,
            verbose=False,
        )
        for result in results_gen:
            yield self._parse_results([result])

    # ── track on a single frame (for live / frame-by-frame) ──

    def track_frame(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack.yaml",
    ) -> list[BBoxRecord]:
        """Run detection + tracking on a single frame.

        Must be called on sequential frames for tracking to work
        (internally the model keeps state when ``persist=True``).
        """
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        return self._parse_results(results)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_results(results) -> list[BBoxRecord]:
        records: list[BBoxRecord] = []
        if not results:
            return records

        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return records

        names = r.names or {}

        for box in boxes:
            # Bug-9: Use squeeze(0) instead of squeeze() for safer dimension handling
            cls_id = int(box.cls.squeeze(0).item())
            conf_val = float(box.conf.squeeze(0).item())
            xyxy = [float(v) for v in box.xyxy.squeeze(0).tolist()]
            track_id = int(box.id.squeeze(0).item()) if box.id is not None else None

            records.append(
                BBoxRecord(
                    class_id=cls_id,
                    class_name=names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf_val,
                    xyxy=xyxy,
                    track_id=track_id,
                )
            )
        return records

    @staticmethod
    def filter_by_role(
        records: list[BBoxRecord],
        student_ids: list[int],
        teacher_ids: list[int],
        env_ids: list[int],
    ) -> tuple[list[BBoxRecord], list[BBoxRecord], list[BBoxRecord]]:
        """Split detections into students / teachers / environment."""
        # Bug-10: Convert to sets for O(1) lookup performance
        s_set = set(student_ids)
        t_set = set(teacher_ids)
        e_set = set(env_ids)
        
        students, teachers, envs = [], [], []
        for r in records:
            if r.class_id in s_set:
                students.append(r)
            elif r.class_id in t_set:
                teachers.append(r)
            elif r.class_id in e_set:
                envs.append(r)
        return students, teachers, envs
