"""YOLO26 detection and ByteTrack multi-object tracking.

This module wraps *ultralytics* YOLO26 for:
  1. Per-frame object detection using a dual-model architecture.
     - Behavior Model: detects 7 student states (SCBehavior).
     - Env Model: detects screen_board.
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
        behavior_weights: str | Path = "yolo26s.pt",
        env_weights: str | Path | None = None,
        device: str = "0",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 960,
    ) -> None:
        self.behavior_model = YOLO(str(behavior_weights))
        self.env_model = YOLO(str(env_weights)) if env_weights else None
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    # ── single-frame detection (no tracking) ──────────────────

    def detect_frame(self, frame: np.ndarray) -> list[BBoxRecord]:
        """Run detection on a single BGR frame and return merged records."""
        results_beh = self.behavior_model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        records = self._parse_results(results_beh)

        if self.env_model:
            results_env = self.env_model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            env_records = self._parse_results(results_env)
            self._merge_env_records(records, env_records)

        return records

    # ── video tracking (ByteTrack) ────────────────────────────

    def track_video(
        self,
        source: str | Path,
        tracker: str = "bytetrack.yaml",
    ) -> Generator[list[BBoxRecord], None, None]:
        """Yield per-frame detection records with persistent track IDs.

        Note: Tracking is applied to the behavior model. Env model (screen_board)
        typically doesn't need tracking.
        """
        results_gen = self.behavior_model.track(
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
        
        # If env_model is present, we must run it per-frame as well.
        # However, .track() returns a generator. We would have to run env_model.predict
        # on the orig_img from the track result.
        for result in results_gen:
            beh_records = self._parse_results([result])
            
            if self.env_model and result.orig_img is not None:
                env_res = self.env_model.predict(
                    source=result.orig_img,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                env_records = self._parse_results(env_res)
                self._merge_env_records(beh_records, env_records)
                
            yield beh_records

    # ── track on a single frame (for live / frame-by-frame) ──

    def track_frame(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack.yaml",
    ) -> list[BBoxRecord]:
        """Run detection + tracking on a single frame."""
        results = self.behavior_model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        records = self._parse_results(results)
        
        if self.env_model:
            env_res = self.env_model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            env_records = self._parse_results(env_res)
            self._merge_env_records(records, env_records)
            
        return records

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_iou(box1: list[float], box2: list[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter_area == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (area1 + area2 - inter_area)

    @classmethod
    def _merge_env_records(cls, beh_records: list[BBoxRecord], env_records: list[BBoxRecord]) -> None:
        """
        Merge environment model records into behavior model records.
        Env model classes (0: student, 1: teacher, 2: screen_board) are shifted to avoid collisions.
        Generic students (class 0) are only added if they don't significantly overlap with 
        an existing behavior model detection (NMS-like box fusion).
        """
        for r in env_records:
            if r.class_id == 1:
                r.class_id = 101
                r.class_name = "teacher"
                beh_records.append(r)
            elif r.class_id == 2:
                r.class_id = 102
                r.class_name = "screen_board"
                beh_records.append(r)
            elif r.class_id == 0:
                # NMS fusion: only add if behavior model completely missed this student
                covered = False
                for br in beh_records:
                    if cls._compute_iou(r.xyxy, br.xyxy) > 0.5:
                        covered = True
                        break
                if not covered:
                    r.class_id = 100
                    r.class_name = "attending"
                    beh_records.append(r)

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
