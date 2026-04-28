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
            # Offset class IDs for env model to avoid collision if necessary
            # Actually, in pipeline.yaml we define env_ids=[2], but the 3-class model outputs:
            # 0:student, 1:teacher, 2:screen_board
            # We ONLY want class 2 from the env model.
            env_records = self._parse_results(results_env)
            records.extend([r for r in env_records if r.class_id == 2])

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
                beh_records.extend([r for r in env_records if r.class_id == 2])
                
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
            records.extend([r for r in env_records if r.class_id == 2])
            
        return records

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
