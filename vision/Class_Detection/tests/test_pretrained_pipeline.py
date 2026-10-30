"""Adapters tested with deterministic observations, independently of model accuracy."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.observation import ContextTimeline, TemporalEvidence
from src.pretrained_pipeline import PretrainedClassroomPipeline, normalize_device, validate_names
from src.schema import KnowledgeAnchor


class Array:
    def __init__(self, values): self.values = np.asarray(values)
    def cpu(self): return self
    def numpy(self): return self.values


def result(boxes, labels=None, ids=None, points=None, names=None):
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=Array(boxes), conf=Array([.9]*len(boxes)),
                              cls=Array(labels or [0]*len(boxes)), id=Array(ids) if ids is not None else None),
        keypoints=SimpleNamespace(data=Array(points)) if points is not None else None,
        names=names or {0:'person'})


class Model:
    def __init__(self, value): self.value = value
    def track(self, *args, **kwargs): return [self.value]
    def predict(self, *args, **kwargs): return [self.value]


def pipeline():
    obj = PretrainedClassroomPipeline.__new__(PretrainedClassroomPipeline)
    pts = np.zeros((17,3)); pts[5]=[30,80,.9]; pts[6]=[70,80,.9]; pts[9]=[25,20,.9]
    obj.pose = Model(result([[0,0,100,200]], ids=[0], points=[pts]))
    obj.behavior = Model(result([[0,0,100,200]], labels=[2], names={2:'writing'}))
    obj.roi=None; obj.device='cpu'; obj.person_conf=.25; obj.imgsz=960; obj.behavior_conf=.4; obj.tracker='unused'
    obj._last_timestamp=None; obj._last_anchor=KnowledgeAnchor(score_k=None,visual_score=None)
    obj._next_frame_id=0
    obj._track_epoch_offset=0; obj._max_emitted_track=-1
    obj._fps=25; obj._last_ocr_time=-100; obj._ocr=None; obj.provenance={'engagement_model':None}
    obj.evidence=TemporalEvidence(min_evidence_sec=.5); obj.context=ContextTimeline()
    return obj


def test_trained_behavior_not_overwritten_by_pose_and_zero_id_is_valid():
    obj=pipeline(); image=np.zeros((240,320,3),dtype=np.uint8)
    for t in [0, .5, 1.]:
        snap=obj.process_frame(image,timestamp_sec=t)
    assert snap.student_states[0].track_id == 0
    assert snap.student_states[0].action.label == 'write'
    assert snap.student_states[0].action.source == 'detection'
    assert snap.student_states[0].cas == .85
    assert snap.knowledge_anchor.score_k is None
    assert snap.classroom_metrics.ctes_score is None
    assert snap.timestamp_sec == 1
    assert snap.data_source == 'pretrained_inference'
    assert snap.student_states[0].gaze.focus_score is None


def test_untracked_detection_never_gets_index_based_identity():
    obj=pipeline(); obj.pose=Model(result([[0,0,100,200]], ids=None, points=[np.zeros((17,3))]))
    snap=obj.process_frame(np.zeros((240,320,3),dtype=np.uint8),timestamp_sec=0)
    assert snap.student_states == []
    assert snap.classroom_metrics.untracked_people == 1
    assert snap.classroom_metrics.mean_cas is None


def test_missing_behavior_and_keypoints_remain_unknown():
    obj=pipeline(); obj.pose=Model(result([[0,0,100,200]], ids=[1], points=[np.zeros((17,3))]))
    obj.behavior=Model(result([]))
    for t in [0,1,2]:
        snap=obj.process_frame(np.zeros((240,320,3),dtype=np.uint8),timestamp_sec=t)
    assert snap.student_states[0].action.label == 'unknown'
    assert snap.student_states[0].cas is None
    assert snap.classroom_metrics.coverage == 0


def test_generic_detection_weights_rejected():
    with pytest.raises(ValueError, match='Generic COCO'):
        validate_names({0:'person',1:'bicycle'}, 'detect')
    validate_names({0:'hand-raising',1:'reading',2:'writing'}, 'detect')


def test_roi_filters_people_before_behavior_assignment():
    obj=pipeline(); obj.roi=[.6,0,1,1]
    snap=obj.process_frame(np.zeros((240,320,3),dtype=np.uint8),timestamp_sec=0)
    assert not snap.student_states


def test_non_monotonic_pipeline_time_rejected():
    obj=pipeline(); image=np.zeros((240,320,3),dtype=np.uint8)
    obj.process_frame(image,timestamp_sec=1)
    with pytest.raises(ValueError, match='timestamps'):
        obj.process_frame(image,timestamp_sec=0)


def test_new_tracking_epoch_cannot_reuse_old_public_ids():
    obj=pipeline(); image=np.zeros((240,320,3),dtype=np.uint8)
    first=obj.process_frame(image,timestamp_sec=0)
    after_gap=obj.process_frame(image,timestamp_sec=10)
    assert first.student_states[0].track_id != after_gap.student_states[0].track_id
    assert after_gap.student_states[0].cas is None


def test_implicit_frame_ids_advance_and_explicit_timestamps_are_preserved():
    obj=pipeline(); image=np.zeros((240,320,3),dtype=np.uint8)
    a=obj.process_frame(image); b=obj.process_frame(image)
    c=obj.process_frame(image,timestamp_sec=1.5)
    assert [a.frame_id,b.frame_id,c.frame_id] == [0,1,2]
    assert [a.timestamp_sec,b.timestamp_sec,c.timestamp_sec] == [0,.04,1.5]


def test_devices_accepted_by_torch_and_ultralytics():
    assert normalize_device('0') == 'cuda:0'
    assert normalize_device('cuda') == 'cuda:0'
    assert normalize_device('cpu') == 'cpu'
    with pytest.raises(ValueError):
        normalize_device('0,1')


def test_unconfirmed_detections_count_even_when_tracker_hides_them():
    obj=pipeline(); image=np.zeros((240,320,3),dtype=np.uint8)
    confirmed=obj.pose.value
    raw=result([[0,0,100,200],[110,0,200,200]],points=[np.zeros((17,3))]*2)
    def tracked(*args,**kwargs):
        obj._capture_pose_detections(SimpleNamespace(results=[raw]))
        return [confirmed]
    obj.pose.track=tracked
    for t in [0,1]: snap=obj.process_frame(image,timestamp_sec=t)
    assert snap.classroom_metrics.active_tracks==1
    assert snap.classroom_metrics.untracked_people==1
    assert snap.classroom_metrics.coverage==.5
