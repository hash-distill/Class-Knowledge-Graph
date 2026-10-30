"""Regression contracts for no-training observations (no model downloads)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.observation import (TemporalEvidence, canonical_behavior, match_behaviors,
                             pose_hand_raise, ContextTimeline, observation_metrics)
from src.schema import StudentState, ActionRecord


def test_generic_coco_is_not_a_classroom_behavior():
    assert canonical_behavior("person") == "unknown"
    assert canonical_behavior("Hand Raising") == "raise_hand"
    assert canonical_behavior("writing") == "write"


def test_assignment_is_one_to_one_and_rejects_far_boxes():
    people = [[0, 0, 100, 200], [40, 0, 140, 200]]
    behaviors = [[5, 5, 95, 195], [500, 0, 600, 200]]
    assert match_behaviors(people, behaviors) == {0: 0}


def test_single_raised_hand_and_invisible_points():
    pts = np.zeros((17, 3))
    pts[5] = [30, 80, .9]; pts[6] = [70, 80, .9]
    pts[9] = [25, 20, .9]; pts[10] = [75, 120, .9]
    assert pose_hand_raise(pts) > .5
    pts[9, 2] = .05
    assert pose_hand_raise(pts) == 0
    assert pose_hand_raise(np.zeros((17, 3))) == 0


def test_time_window_sampling_independence():
    outputs = []
    for step in [.2, 1.0]:
        stream = TemporalEvidence(window_sec=4, min_evidence_sec=.5)
        for t in np.arange(0, 4.01, step):
            stream.push(1, float(t), 'write', .9, .85, 'lecture')
        score, report = stream.assess(1, 4, 'lecture')
        outputs.append(score)
        assert report.temporal_coverage == pytest.approx(1)
    assert outputs == pytest.approx([.85, .85])


def test_unknown_is_not_carried_or_counted_as_zero():
    stream = TemporalEvidence(min_evidence_sec=.1)
    stream.push(1, 0, 'read', .8, .8, 'lecture')
    stream.push(1, 1, 'read', .8, .8, 'lecture')
    assert stream.assess(1, 1, 'lecture')[0] is not None
    stream.push(1, 2, 'unknown', 0, None, 'lecture')
    assert stream.assess(1, 2, 'lecture')[0] is None
    students = [StudentState(track_id=1, bbox=[0,0,1,1], cas=None),
                StudentState(track_id=2, bbox=[0,0,1,1], cas=.8)]
    metrics = observation_metrics(students)
    assert metrics.mean_cas == .8
    assert metrics.coverage == .5
    assert metrics.ctes_score is None
    assert observation_metrics([]).mean_cas is None


def test_history_expires_by_seconds_and_context_does_not_bleed():
    stream = TemporalEvidence(window_sec=4, retention_sec=6)
    stream.push(1, 0, 'write', .9, .85, 'lecture')
    stream.prune(1); assert 1 in stream.tracks
    stream.push(1, 1, 'write', .9, .95, 'exercise')
    assert stream.assess(1, 1, 'exercise')[0] is None
    stream.prune(8); assert 1 not in stream.tracks


def test_non_monotonic_samples_rejected():
    stream = TemporalEvidence()
    stream.push(1, 2, 'read', .8, .8, 'lecture')
    with pytest.raises(ValueError, match='timestamp'):
        stream.push(1, 1, 'read', .8, .8, 'lecture')


def test_context_half_open_segments_and_overlap_validation():
    context = ContextTimeline([{'start_sec':0,'end_sec':3,'task':'lecture','knowledge_point':'排列'},
                               {'start_sec':3,'end_sec':5,'task':'exercise'}])
    assert context.at(2)['task'] == 'lecture'
    assert context.at(3)['task'] == 'exercise'
    assert context.at(6)['task'] == 'unspecified'
    with pytest.raises(ValueError):
        ContextTimeline([{'start_sec':0,'end_sec':3,'task':'lecture'},
                         {'start_sec':2,'end_sec':5,'task':'exercise'}])
