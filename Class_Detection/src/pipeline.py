"""End-to-end classroom evaluation pipeline.

Orchestrates all CV modules into a single ``process_frame`` call:
  Detection → Tracking → Pose → Action → Gaze → VSAM → CAS/CTES → JSON
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.schema import (
    ActionRecord,
    ClassroomSnapshot,
    GazeRecord,
    KnowledgeAnchor,
    StudentState,
)
from src.detector import Detector
from src.pose import PoseEstimator
from src.action import ActionClassifier
from src.gaze import GazeEstimator
from src.ocr_anchor import OCRAnchorDetector
from src.vsam import VSAMAligner
from src.scoring import calc_cas, compute_classroom_metrics


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _compute_iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

class ClassroomPipeline:
    """Full pipeline: frame in → ClassroomSnapshot out."""

    def __init__(
        self,
        config_path: str | Path = "configs/pipeline.yaml",
        det_weights: str | None = None,
        pose_weights: str | None = None,
        device: str | None = None,
    ) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        # ── detection ─────────────────────────────────────────
        det_cfg = cfg.get("detection", {})
        run_device = device if device is not None else det_cfg.get("device", "0")
        self.detector = Detector(
            weights=det_weights if det_weights is not None else det_cfg.get("model", "yolo26s.pt"),
            device=run_device,
            conf=det_cfg.get("conf_threshold", 0.25),
            iou=det_cfg.get("iou_threshold", 0.7),
            imgsz=det_cfg.get("imgsz", 960),
        )

        # ── pose ──────────────────────────────────────────────
        pose_cfg = cfg.get("pose", {})
        self.pose = PoseEstimator(
            weights=pose_weights if pose_weights is not None else pose_cfg.get("model", "yolo26n-pose.pt"),
            device=run_device,
            conf=pose_cfg.get("conf_threshold", 0.3),
        )

        # ── action ────────────────────────────────────────────
        act_cfg = cfg.get("action", {})
        self.action = ActionClassifier(
            use_stgcn=act_cfg.get("use_stgcn", False),
            stgcn_weights=act_cfg.get("stgcn_weights"),
            window_size=act_cfg.get("window_size", 30),
            action_scores=act_cfg.get("action_scores"),
            device=run_device,
        )

        # ── gaze ──────────────────────────────────────────────
        gaze_cfg = cfg.get("gaze", {})
        self.gaze = GazeEstimator(
            min_face_conf=pose_cfg.get("min_face_kpt_conf", 0.4),
            focus_zones=gaze_cfg.get("focus_zones"),
            fallback_base=gaze_cfg.get("fallback_base_score", 0.55),
        )

        # ── OCR ───────────────────────────────────────────────
        vsam_cfg = cfg.get("vsam", {})
        self.ocr = OCRAnchorDetector(
            change_threshold=vsam_cfg.get("text_change_threshold", 0.3),
        )
        self.ocr_interval = vsam_cfg.get("ocr_interval_frames", 30)

        # ── VSAM ──────────────────────────────────────────────
        self.vsam = VSAMAligner(
            mu=vsam_cfg.get("mu", 3.0),
            sigma=vsam_cfg.get("sigma", 1.5),
            window_duration=vsam_cfg.get("window_duration", 12.0),
        )

        # ── scoring ───────────────────────────────────────────
        score_cfg = cfg.get("scoring", {})
        self.w_action = score_cfg.get("w_action", 0.6) # updated default
        self.w_gaze = score_cfg.get("w_gaze", 0.4)     # updated default
        self.lambda_penalty = score_cfg.get("lambda_penalty", 1.0)

        # ── role class IDs ────────────────────────────────────
        # Default now matches the merged 5-class model mapping
        self.student_ids = cfg.get("student_class_ids", [0, 1, 2])
        self.teacher_ids = cfg.get("teacher_class_ids", [3])
        self.env_ids = cfg.get("environment_class_ids", [4])

        # ── runtime state ────────────────────────────────────
        self._frame_count: int = 0
        self._fps: float = 25.0

    def set_fps(self, fps: float) -> None:
        self._fps = max(fps, 1.0)

    # ── main entry point ─────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> ClassroomSnapshot:
        """Process a single BGR frame through the full pipeline.
        
        If frame_id is provided, it is used for timing; otherwise, 
        internal self._frame_count is used.
        """
        if frame_id is not None:
            self._frame_count = frame_id
        else:
            self._frame_count += 1
            
        t_sec = self._frame_count / self._fps

        # 1. Detection + Tracking
        detections = self.detector.track_frame(frame)
        students, teachers, envs = Detector.filter_by_role(
            detections, self.student_ids, self.teacher_ids, self.env_ids,
        )

        # 2. Pose estimation (whole frame)
        kpt_records, pose_bboxes = self.pose.estimate(frame)
        h, w = frame.shape[:2]

        # 3. Per-student processing
        student_states: list[StudentState] = []
        for i, det in enumerate(students):
            tid = det.track_id or (10000 + i)

            # Match keypoint to detection by bbox IoU
            best_iou = 0.0
            best_kpt_idx = -1
            for j, p_box in enumerate(pose_bboxes):
                iou = _compute_iou(det.xyxy, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_kpt_idx = j
                    
            kpt = kpt_records[best_kpt_idx] if best_iou > 0.3 else None

            # Action
            action_rec = self.action.classify_from_detection(
                det.class_name, det.confidence,
            )
            if kpt is not None:
                self.action.push_keypoints(tid, kpt.points)
                kpt_action = self.action.classify_from_keypoints(tid)
                if kpt_action is not None:
                    action_rec = kpt_action

            # Gaze
            gaze_rec = GazeRecord()
            if kpt is not None:
                gaze_rec = self.gaze.estimate(kpt, frame_shape=(h, w))

            # CAS
            s_action = action_rec.engagement_score
            s_gaze = gaze_rec.focus_score
            cas = calc_cas(s_action, s_gaze, action_rec.label, self.w_action, self.w_gaze)

            student_states.append(
                StudentState(
                    track_id=tid,
                    bbox=det.xyxy,
                    action=action_rec,
                    gaze=gaze_rec,
                    cas=round(cas, 4),
                )
            )

        # Bug-7: Prune action buffer for stale tracks
        active_tids = {s.track_id for s in student_states}
        self.action.buffer.prune(active_tids)

        # 4. OCR on environment regions
        if self._frame_count % self.ocr_interval == 0:
            for env_det in envs:
                x1, y1, x2, y2 = [int(v) for v in env_det.xyxy]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 > 50 and y2 - y1 > 50:
                    crop = frame[y1:y2, x1:x2]
                    new_text = self.ocr.detect_change(crop)
                    if new_text is not None:
                        self.vsam.trigger(entity=new_text[:60], t_ocr=t_sec)

        # 5. VSAM feed
        metrics = compute_classroom_metrics(student_states, self.lambda_penalty)
        self.vsam.feed(t_sec, metrics.mean_cas)
        closed = self.vsam.evaluate(t_sec)

        # Build knowledge anchor from most recent closed event
        anchor = KnowledgeAnchor()
        if closed:
            last = closed[-1]
            anchor = KnowledgeAnchor(
                entity=last.entity,
                trigger_time=f"{last.t_ocr:.2f}s",
                gaussian_weight=round(
                    max(
                        (1.0 if not last.cas_buffer else
                         max(last.cas_buffer) - min(last.cas_buffer)),
                        0.0,
                    ),
                    4,
                ),
                score_k=round(last.score_k, 4),
            )

        return ClassroomSnapshot(
            timestamp=_utc_iso(),
            frame_id=self._frame_count,
            knowledge_anchor=anchor,
            classroom_metrics=metrics,
            student_states=student_states,
        )
