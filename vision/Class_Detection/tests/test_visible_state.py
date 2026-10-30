"""Geometry contracts, not classroom accuracy measurements."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visible_state import visible_state


def points():
    p = np.zeros((17, 3))
    p[0] = [50, 40, .9]; p[1] = [45, 35, .9]; p[2] = [55, 35, .9]
    p[5] = [20, 80, .95]; p[6] = [80, 80, .95]
    return p


def test_visible_head_state_is_additive_and_model_action_remains_primary():
    result = visible_state(points(), 'write', .85, 'pretrained_detector')
    assert result['student_state']['label'] == 'write'
    assert result['student_state']['source'] == 'pretrained_detector'
    assert result['posture_tags'][0]['label'] == 'head_up'
    assert result['posture_tags'][0]['metrics']['nose_shoulder_gap_ratio'] > .28


def test_lowered_head_requires_current_confident_face_and_shoulders():
    p = points(); p[0, 1] = 82
    result = visible_state(p)
    assert result['student_state']['label'] == 'head_lowered'
    assert result['student_state']['source'] == 'pose_geometry_rule'
    p[0, 2] = .1
    assert visible_state(p)['student_state']['label'] == 'unknown'
    p[0, 2] = .9; p[5, 2] = .1
    assert visible_state(p)['student_state']['label'] == 'unknown'


def test_ambiguous_head_angle_does_not_get_forced_into_a_state():
    p = points(); p[0, 1] = 67
    assert visible_state(p)['student_state']['label'] == 'unknown'
    assert visible_state(np.full((17, 3), float('nan')))['student_state']['label'] == 'unknown'


def test_hand_near_head_requires_visible_wrist_and_does_not_mean_phone_or_sleep():
    p = points(); p[9] = [52, 42, .9]
    result = visible_state(p)
    assert 'hand_near_head' in [tag['label'] for tag in result['posture_tags']]
    assert result['student_state']['label'] == 'hand_near_head'
    assert not any(word in str(result) for word in ('phone', 'sleep'))
    p[9, 2] = .1
    assert 'hand_near_head' not in [tag['label'] for tag in visible_state(p)['posture_tags']]


def test_seated_or_occluded_lower_body_never_becomes_standing():
    p = points()
    assert 'standing' not in [tag['label'] for tag in visible_state(p)['posture_tags']]
    for sh, hip, knee, ankle, x in [(5,11,13,15,20),(6,12,14,16,80)]:
        p[hip]=[x,140,.9]; p[knee]=[x+60,140,.9]; p[ankle]=[x+60,210,.9]
    assert 'standing' not in [tag['label'] for tag in visible_state(p)['posture_tags']]
    for hip,knee,ankle,x in [(11,13,15,20),(12,14,16,80)]:
        p[hip]=[x,140,.9]; p[knee]=[x,200,.9]; p[ankle]=[x,260,.9]
    assert 'standing' in [tag['label'] for tag in visible_state(p)['posture_tags']]
