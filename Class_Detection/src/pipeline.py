"""End-to-end classroom evaluation pipeline.

Supports both:
1. Single-stream mode: one classroom video contains students and screen.
2. Dual-stream mode: a student video and a PPT/screen video are aligned on
   the same timeline through explicit timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.action import ActionClassifier
from src.detector import Detector
from src.gaze import GazeEstimator
from src.ocr_anchor import OCRAnchorDetector
from src.pose import PoseEstimator
from src.schema import (
    ClassroomSnapshot,
    GazeRecord,
    KnowledgeAnchor,
    StudentState,
)
from src.scoring import calc_cas, compute_classroom_metrics
from src.vsam import VSAMAligner, score_knowledge_point


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(box_a_area + box_b_area - inter_area)


def _clip_crop(frame: np.ndarray, crop_box: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop_box is None:
        return frame

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = crop_box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return frame

    return frame[y1:y2, x1:x2]


class ClassroomPipeline:
    """Full classroom pipeline with reusable stateful VSAM alignment."""

    def __init__(
        self,
        config_path: str | Path = "configs/pipeline.yaml",
        det_weights: str | None = None,
        env_weights: str | None = None,
        pose_weights: str | None = None,
        device: str | None = None,
    ) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        det_cfg = cfg.get("detection", {})
        run_device = device if device is not None else det_cfg.get("device", "0")
        behavior_weights = det_weights if det_weights is not None else det_cfg.get("behavior_model", "yolo26s.pt")
        final_env_weights = env_weights if env_weights is not None else det_cfg.get("env_model")

        self.detector = Detector(
            behavior_weights=behavior_weights,
            env_weights=final_env_weights,
            device=run_device,
            conf=det_cfg.get("conf_threshold", 0.25),
            iou=det_cfg.get("iou_threshold", 0.7),
            imgsz=det_cfg.get("imgsz", 960),
        )

        pose_cfg = cfg.get("pose", {})
        self.pose = PoseEstimator(
            weights=pose_weights if pose_weights is not None else pose_cfg.get("model", "yolo26n-pose.pt"),
            device=run_device,
            conf=pose_cfg.get("conf_threshold", 0.3),
        )

        track_cfg = cfg.get("tracking", {})
        self.tracker_path = track_cfg.get("tracker", "bytetrack.yaml")

        act_cfg = cfg.get("action", {})
        self.action = ActionClassifier(
            use_stgcn=act_cfg.get("use_stgcn", False),
            stgcn_weights=act_cfg.get("stgcn_weights"),
            window_size=act_cfg.get("window_size", 30),
            action_scores=act_cfg.get("action_scores"),
            device=run_device,
        )

        gaze_cfg = cfg.get("gaze", {})
        self.gaze = GazeEstimator(
            min_face_conf=pose_cfg.get("min_face_kpt_conf", 0.4),
            focus_zones=gaze_cfg.get("focus_zones"),
            fallback_base=gaze_cfg.get("fallback_base_score", 0.55),
        )

        vsam_cfg = cfg.get("vsam", {})
        self.ocr = OCRAnchorDetector(
            change_threshold=vsam_cfg.get("text_change_threshold", 0.3),
        )
        self.ocr_interval = vsam_cfg.get("ocr_interval_frames", 30)

        self.vsam = VSAMAligner(
            mu=vsam_cfg.get("mu", 3.0),
            sigma=vsam_cfg.get("sigma", 1.5),
            window_duration=vsam_cfg.get("window_duration", 12.0),
        )

        score_cfg = cfg.get("scoring", {})
        self.w_action = score_cfg.get("w_action", 0.6)
        self.w_gaze = score_cfg.get("w_gaze", 0.4)
        self.lambda_penalty = score_cfg.get("lambda_penalty", 1.0)

        self.student_ids = cfg.get("student_class_ids", [0, 1, 2])
        self.teacher_ids = cfg.get("teacher_class_ids", [3])
        self.env_ids = cfg.get("environment_class_ids", [4])

        self._frame_count: int = 0
        self._fps: float = 25.0
        self._last_ocr_time: float = -999.0

    def set_fps(self, fps: float) -> None:
        self._fps = max(fps, 1.0)

    def register_anchor(self, entity: str, timestamp_sec: float) -> None:
        entity = entity.strip()
        if not entity:
            return
        self.vsam.trigger(entity=entity[:60], t_ocr=float(timestamp_sec))

    def process_anchor_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Optional[str]:
        crop = _clip_crop(frame, crop_box)
        detected_text = self.ocr.detect_change(crop)
        if detected_text is None:
            return None

        entity = detected_text[:60].strip()
        if not entity:
            return None

        self.register_anchor(entity, timestamp_sec)
        return entity

    def process_student_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
        ppt_crop: tuple[int, int, int, int] | None = None,
    ) -> ClassroomSnapshot:
        return self.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            enable_ocr=False,
            ppt_crop=ppt_crop,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
        enable_ocr: bool = True,
        ppt_crop: tuple[int, int, int, int] | None = None,
    ) -> ClassroomSnapshot:
        """Process one frame.

        `timestamp_sec` allows dual-stream mode to place student and PPT events
        on the same shared timeline.
        """
        if frame_id is not None:
            self._frame_count = frame_id
        else:
            self._frame_count += 1

        t_sec = float(timestamp_sec) if timestamp_sec is not None else self._frame_count / self._fps

        detections = self.detector.track_frame(frame, tracker=self.tracker_path)
        students, _teachers, envs = Detector.filter_by_role(
            detections, self.student_ids, self.teacher_ids, self.env_ids,
        )

        keypoint_records, pose_bboxes = self.pose.estimate(frame)
        height, width = frame.shape[:2]

        student_states: list[StudentState] = []
        for index, det in enumerate(students):
            track_id = det.track_id or (10000 + index)

            best_iou = 0.0
            best_keypoint_index = -1
            for pose_index, pose_box in enumerate(pose_bboxes):
                iou = _compute_iou(det.xyxy, pose_box)
                if iou > best_iou:
                    best_iou = iou
                    best_keypoint_index = pose_index

            keypoints = keypoint_records[best_keypoint_index] if best_iou > 0.3 else None

            action_record = self.action.classify_from_detection(
                det.class_name,
                det.confidence,
            )
            if keypoints is not None:
                self.action.push_keypoints(track_id, keypoints.points)
                keypoint_action = self.action.classify_from_keypoints(track_id)
                if keypoint_action is not None:
                    action_record = keypoint_action

            gaze_record = GazeRecord()
            if keypoints is not None:
                gaze_record = self.gaze.estimate(keypoints, frame_shape=(height, width))

            cas = calc_cas(
                action_record.engagement_score,
                gaze_record.focus_score,
                action_record.label,
                self.w_action,
                self.w_gaze,
            )

            student_states.append(
                StudentState(
                    track_id=track_id,
                    bbox=det.xyxy,
                    action=action_record,
                    gaze=gaze_record,
                    cas=round(cas, 4),
                )
            )

        active_track_ids = {student.track_id for student in student_states}
        self.action.buffer.prune(active_track_ids)

        if enable_ocr:
            ocr_interval_sec = self.ocr_interval / self._fps
            if t_sec - self._last_ocr_time >= ocr_interval_sec:
                ocr_triggered = False
                
                if ppt_crop is not None:
                    crop = _clip_crop(frame, ppt_crop)
                    detected_text = self.ocr.detect_change(crop)
                    if detected_text is not None:
                        self.register_anchor(detected_text, t_sec)
                    ocr_triggered = True
                else:
                    for env_det in envs:
                        x1, y1, x2, y2 = [int(value) for value in env_det.xyxy]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        if x2 - x1 <= 50 or y2 - y1 <= 50:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        detected_text = self.ocr.detect_change(crop)
                        if detected_text is not None:
                            self.register_anchor(detected_text, t_sec)
                        ocr_triggered = True
                        break  # Only OCR the first valid screen box

                if ocr_triggered:
                    self._last_ocr_time = t_sec

        metrics = compute_classroom_metrics(student_states, self.lambda_penalty)
        self.vsam.feed(t_sec, metrics.mean_cas)
        self.vsam.evaluate(t_sec)

        return ClassroomSnapshot(
            timestamp=_utc_iso(),
            frame_id=self._frame_count,
            knowledge_anchor=self._build_current_anchor(),
            classroom_metrics=metrics,
            student_states=student_states,
            env_bboxes=[env.xyxy for env in envs],
        )

    def _build_current_anchor(self) -> KnowledgeAnchor:
        anchors = self.vsam.all_anchors
        if not anchors:
            return KnowledgeAnchor()

        anchor_event = anchors[-1]
        if anchor_event.closed:
            score_value = anchor_event.score_k
        else:
            score_value = score_knowledge_point(
                anchor_event.cas_buffer,
                anchor_event.time_buffer,
                anchor_event.t_ocr,
                self.vsam.mu,
                self.vsam.sigma,
            )

        # Compute the actual Gaussian alignment weight at the latest
        # observation time (how well the current moment aligns with
        # the expected student reaction window).
        from src.vsam import gaussian_weight as _gw

        if anchor_event.time_buffer:
            latest_t = anchor_event.time_buffer[-1]
        else:
            latest_t = anchor_event.t_ocr
        gw_value = round(_gw(latest_t, anchor_event.t_ocr, self.vsam.mu, self.vsam.sigma), 4)

        rounded_score = round(score_value, 4)
        return KnowledgeAnchor(
            entity=anchor_event.entity,
            trigger_time=f"{anchor_event.t_ocr:.2f}s",
            gaussian_weight=gw_value,
            score_k=rounded_score,
            visual_score=rounded_score,
        )
