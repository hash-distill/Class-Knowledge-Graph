"""Pretrained observation pipeline. No classroom engagement model is trained here."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

from src.observation import (CONTEXT_SCORES, ContextTimeline, TemporalEvidence,
                             canonical_behavior, keypoint_quality, match_behaviors,
                             observation_metrics, pose_hand_raise)
from src.schema import (ActionRecord, ActionSource, ClassroomSnapshot, FocusZone,
                        GazeRecord, KnowledgeAnchor, StudentState)
from src.visible_state import visible_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT/value).resolve()


def validate_names(names, task):
    values = list(names.values()) if isinstance(names, dict) else list(names)
    if task == 'pose':
        if values != ['person']:
            raise ValueError(f'Expected a COCO person pose model; actual classes: {values}')
    elif not values or any(canonical_behavior(x) == 'unknown' for x in values):
        raise ValueError(f'Expected classroom read/write/raise-hand weights; actual classes: {values}. '
                         'Generic COCO detection weights cannot classify classroom behavior.')


def normalize_device(device):
    value = str(device).strip().lower()
    if value.isdecimal():
        return f'cuda:{value}'
    if value == 'cuda':
        return 'cuda:0'
    if value == 'cpu' or (value.startswith('cuda:') and value[5:].isdecimal()):
        return value
    raise ValueError('Device must be auto, cpu, cuda:N or a single GPU number')


def load_model(path, key, manifest, device):
    """Verify pinned bytes before deserialization; overrides remain explicitly identified."""
    from scripts.fetch_pretrained import verify_file
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing weights: {path}. Run scripts/fetch_pretrained.py first.')
    spec = manifest[key]
    pinned = path.name == spec['filename']
    if pinned:
        verify_file(path, spec)
        digest = spec['sha256']
    else:
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT/'artifacts'/'ultralytics'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from ultralytics import YOLO
    model = YOLO(str(path))
    expected_task = 'pose' if key == 'pose_coco' else 'detect'
    if model.task != expected_task:
        raise ValueError(f'{path.name}: expected {expected_task}, got {model.task}')
    validate_names(model.names, expected_task)
    actual_names = {str(k): v for k, v in model.names.items()}
    architecture = Path(model.model.yaml.get('yaml_file', '')).name
    if pinned and spec.get('expected_names') and actual_names != spec['expected_names']:
        raise ValueError(f'{path.name}: class IDs/names differ from pinned manifest')
    if pinned and spec.get('checkpoint_architecture') and architecture != spec['checkpoint_architecture']:
        raise ValueError(f'{path.name}: architecture differs from pinned manifest')
    if expected_task == 'pose' and list(model.model.yaml.get('kpt_shape', [])) != [17, 3]:
        raise ValueError('Pose weights must emit 17 COCO keypoints with confidence')
    model.to(normalize_device(device))
    return model, {'file': path.name, 'sha256': digest, 'task': expected_task,
                   'architecture': architecture,
                   'names': model.names, 'pinned': pinned,
                   'source_url': spec['source_url'] if pinned else 'user_supplied_local_weights',
                   'revision': spec['revision'] if pinned else None}


class PretrainedClassroomPipeline:
    def __init__(self, config_path, det_weights=None, pose_weights=None, device=None,
                 context_path=None, task=None, student_roi=None, pose_only=False):
        self.config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        cfg = self.config
        import torch
        requested_device = str(device or cfg.get('device', 'auto'))
        self.device = normalize_device(('cuda:0' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device)
        if self.device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Use --device cpu or install a CUDA PyTorch build.')
        if self.device == 'cpu':
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        from scripts.fetch_pretrained import load_manifest
        manifest = load_manifest()
        models = cfg['models']
        self.pose, pose_meta = load_model(pose_weights or models['pose'], 'pose_coco', manifest, self.device)
        # Register before the first .track(): Ultralytics appends its tracking callback later.
        # Tracking hides unconfirmed detections, which must still count toward coverage.
        self._raw_person_boxes = None
        self.pose.add_callback('on_predict_postprocess_end', self._capture_pose_detections)
        self.behavior = None
        provenance_models = [pose_meta]
        behavior_path = det_weights or models.get('behavior')
        if behavior_path and not pose_only:
            self.behavior, behavior_meta = load_model(behavior_path, 'behavior_scb', manifest, self.device)
            provenance_models.append(behavior_meta)
        self.provenance = {'pipeline': 'pretrained_observation_v1', 'models': provenance_models,
                           'engagement_model': None, 'score_semantics': 'uncalibrated_behavior_rule',
                           'supported_behaviors': ['read', 'write', 'raise_hand'] if self.behavior else [],
                           'device': self.device}
        self.provenance['visible_states'] = {'method':'coco17_geometry_v1', 'trained':False,
            'posture_scores':None, 'scope':'current_source_frame', 'role_inference':False,
            'supported_tags':['head_up','head_lowered','hand_near_head','raise_hand','standing']}
        self.roi = student_roi or cfg.get('student_roi')
        if self.roi is not None:
            if len(self.roi) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in self.roi) or not (self.roi[0] < self.roi[2] and self.roi[1] < self.roi[3]):
                raise ValueError('student_roi must be normalized x1,y1,x2,y2 within [0,1]')
        self.imgsz = int(cfg.get('imgsz', 960))
        self.person_conf = float(cfg.get('person_conf', .25))
        self.behavior_conf = float(cfg.get('behavior_conf', .4))
        self.tracker = str(resolve_project_path(cfg.get('tracker', 'configs/bytetrack_low.yaml')))
        self.evidence = TemporalEvidence(**cfg.get('temporal', {}))
        self.context = ContextTimeline(
            json.loads(Path(context_path).read_text(encoding='utf-8')) if context_path else [],
            default_task=task or cfg.get('default_task', 'unspecified'))
        self._last_timestamp = None
        self._next_frame_id = 0
        self._track_epoch_offset = 0
        self._max_emitted_track = -1
        self._last_anchor = KnowledgeAnchor(score_k=None, visual_score=None)
        self._fps = 25.
        self._ocr = None
        self._last_ocr_time = -math.inf

    def set_fps(self, fps):
        self._fps = max(float(fps), 1.)

    def register_anchor(self, entity, timestamp_sec):
        self._last_anchor = KnowledgeAnchor(entity=entity.strip()[:120],
                                            trigger_time=str(timestamp_sec), score_k=None, visual_score=None)

    def process_student_frame(self, frame, frame_id=None, timestamp_sec=None, **kwargs):
        return self.process_frame(frame, frame_id, timestamp_sec, enable_ocr=False)

    def _capture_pose_detections(self, predictor):
        boxes = predictor.results[0].boxes if predictor.results else None
        self._raw_person_boxes = boxes.xyxy.cpu().numpy().copy() if boxes is not None else np.empty((0,4))

    def _inside_roi(self, box, width, height):
        cx, cy = (box[0]+box[2])/(2*width), (box[1]+box[3])/(2*height)
        return not self.roi or (self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3])

    def process_frame(self, frame, frame_id=None, timestamp_sec=None, enable_ocr=False, ppt_crop=None):
        effective_frame_id = self._next_frame_id if frame_id is None else int(frame_id)
        if effective_frame_id < 0:
            raise ValueError('Frame ID must be nonnegative')
        timestamp = float(timestamp_sec if timestamp_sec is not None else effective_frame_id/self._fps)
        if not math.isfinite(timestamp) or timestamp < 0 or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError('Frame timestamps must be finite and strictly increasing')
        # A seek / long absence starts new tracking state; histories cannot leak across gaps.
        if self._last_timestamp is not None and timestamp-self._last_timestamp > self.evidence.retention_sec:
            for tracker in getattr(getattr(self.pose, 'predictor', None), 'trackers', []):
                tracker.reset()
            self.evidence.tracks.clear()
            self._track_epoch_offset = self._max_emitted_track + 1
        self._last_timestamp = timestamp
        self._next_frame_id = effective_frame_id + 1
        self._raw_person_boxes = None
        result = self.pose.track(frame, persist=True, tracker=self.tracker, conf=self.person_conf,
                                 imgsz=self.imgsz, device=self.device, verbose=False)[0]
        h, w = frame.shape[:2]
        people, points, ids, confidences = [], [], [], []
        untracked = 0
        if result.boxes is not None:
            box_data = result.boxes.xyxy.cpu().numpy()
            kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
            det_conf = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(box_data):
                if not self._inside_roi(box, w, h):
                    continue
                if track_ids is None:
                    untracked += 1
                    continue
                public_id = int(track_ids[i]) + self._track_epoch_offset
                self._max_emitted_track = max(self._max_emitted_track, public_id)
                people.append(box.tolist()); ids.append(public_id); confidences.append(float(det_conf[i]))
                points.append(kpts[i] if kpts is not None and i < len(kpts) else np.zeros((17,3)))
        if self._raw_person_boxes is not None:
            raw_count = sum(self._inside_roi(box, w, h) for box in self._raw_person_boxes)
            untracked = max(raw_count-len(people), 0)
        behavior_boxes, behavior_labels, behavior_confidences = [], [], []
        if self.behavior is not None and people:
            behavior_result = self.behavior.predict(frame, conf=self.behavior_conf, imgsz=self.imgsz,
                                                    device=self.device, verbose=False)[0]
            if behavior_result.boxes is not None:
                behavior_boxes = behavior_result.boxes.xyxy.cpu().numpy().tolist()
                behavior_labels = [canonical_behavior(behavior_result.names[int(i)]) for i in behavior_result.boxes.cls.cpu().numpy()]
                behavior_confidences = behavior_result.boxes.conf.cpu().numpy().tolist()
        matches = match_behaviors(people, behavior_boxes)
        context = self.context.at(timestamp)
        task = context['task']
        students = []
        for i, (box, tid, keypoints) in enumerate(zip(people, ids, points)):
            label, confidence, source = 'unknown', 0., ActionSource.RULE
            quality = keypoint_quality(keypoints)
            box_quality = min(1., max(0., (box[3]-box[1])/80))
            if i in matches:
                j = matches[i]
                label, confidence, source = behavior_labels[j], float(behavior_confidences[j]), ActionSource.DETECTION
                observation_quality = confidence * box_quality
            else:
                raised = pose_hand_raise(keypoints)
                if raised > 0:
                    label, confidence = 'raise_hand', raised
                observation_quality = confidence * quality * box_quality
            score = CONTEXT_SCORES[task].get(label) if observation_quality >= self.evidence.min_quality else None
            self.evidence.push(tid, timestamp, label, observation_quality, score, task)
            cas, assessment = self.evidence.assess(tid, timestamp, task)
            if source == ActionSource.RULE and label != 'unknown':
                assessment.reasons.append('举手来自可见腕部/肩部几何规则，非课堂行为模型')
            students.append(StudentState(
                track_id=tid, bbox=box, cas=cas, assessment=assessment, keypoint_quality=round(quality,4),
                action=ActionRecord(label=label, confidence=round(confidence,4), engagement_score=score,
                                    det_confidence=round(confidence if source == ActionSource.DETECTION else 0.,4), source=source),
                gaze=GazeRecord(focus_score=None, focus_zone=FocusZone.UNKNOWN),
                keypoints=np.round(keypoints,4).tolist(),
                **visible_state(keypoints, label, confidence,
                    'pretrained_detector' if source == ActionSource.DETECTION else 'pose_geometry_rule'),
            ))
        self.evidence.prune(timestamp)
        if enable_ocr and ppt_crop and timestamp-self._last_ocr_time >= 1:
            if self._ocr is None:
                from src.ocr_anchor import OCRAnchorDetector
                self._ocr = OCRAnchorDetector()
            x1,y1,x2,y2 = ppt_crop
            crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size:
                text = self._ocr.detect_change(crop)
                if text:
                    self.register_anchor(text, timestamp)
            self._last_ocr_time = timestamp
        anchor = self._last_anchor.model_copy(deep=True)
        if context.get('knowledge_point'):
            anchor = KnowledgeAnchor(entity=str(context['knowledge_point']), trigger_time=str(context['start_sec']),
                                     score_k=None, visual_score=None)
        return ClassroomSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(), timestamp_sec=timestamp,
            frame_id=effective_frame_id, schema_version='2.0', data_source='pretrained_inference',
            population_scope='visible_people_in_roi', provenance=self.provenance,
            student_states=students, classroom_metrics=observation_metrics(students, untracked),
            knowledge_anchor=anchor,
        )
