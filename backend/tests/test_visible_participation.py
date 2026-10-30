"""Current-frame denominators and abstention must remain auditable."""
import copy
import pytest
from review.activity import visible_participation, validate_activity, add_visible_summary, frame_evidence


def person(label, source='pose_geometry_rule'):
    return {'student_state': {'label': label, 'name': label, 'source': source,
            'confidence': .9, 'evidence': ['test geometry'], 'keypoint_indices': [0,5,6]}}


def test_current_frame_summary_separates_active_actions_from_posture():
    row = {'participants':[person('write','pretrained_detector'),person('head_up'),person('unknown')],
           'detected_people':4}
    result = visible_participation(row)
    assert result['observed_people'] == 2 and result['unknown_people'] == 2
    assert result['coverage'] == .5 and result['active_people'] == 1
    assert result['active_fraction'] == .25
    assert result['index'] == 85 and result['score_coverage'] == .25
    assert result['weights']['write'] == .85 and 'head_up' not in result['weights']


def test_no_people_and_unknown_remain_unavailable_not_zero():
    for row in [{'participants':[],'detected_people':0},
                {'participants':[person('unknown')],'detected_people':1}]:
        result = visible_participation(row)
        assert result['index'] is None
    assert visible_participation({'participants':[],'detected_people':0})['coverage'] is None
    assert visible_participation({'participants':[person('unknown')],'detected_people':1})['active_fraction'] == 0


def test_unverified_standing_is_displayed_but_never_scored_as_active():
    result = visible_participation({'participants':[person('standing')],'detected_people':1})
    assert result['observed_people'] == 1 and result['index'] is None
    assert result['active_people'] == 0 and result['scored_people'] == 0


def test_head_posture_never_receives_a_participation_score():
    for label in ['head_up','head_lowered','hand_near_head']:
        result = visible_participation({'participants':[person(label)],'detected_people':1})
        assert result['observed_people'] == 1 and result['index'] is None
        assert result['score_coverage'] == 0


def test_enriched_validation_checks_counts_keypoint_support_and_model_provenance(tmp_path):
    from test_activity import fixture
    bundle, activity = fixture(tmp_path)
    for row in activity['frames']:
        p = row['participants'][0]
        p.update(person('head_up'))
        p['keypoints'] = [[5.,5.,.9] for _ in range(17)]
        p['posture_tags'] = [copy.deepcopy(p['student_state'])]
        add_visible_summary(row)
    assert validate_activity(activity,bundle,tmp_path)
    for mutate in [
        lambda a:a['frames'][0]['participation'].update(index=65),
        lambda a:a['frames'][0]['student_state_counts'].update(head_up=2),
        lambda a:a['frames'][0]['participants'][0]['keypoints'][0].__setitem__(2,.1),
        lambda a:a['frames'][0]['participants'][0]['student_state'].update(label='sleep'),
        lambda a:a['frames'][0]['participants'][0]['posture_tags'][0].update(source='guessed'),
        lambda a:a['frames'][0]['participants'][0]['student_state'].update(label='read',source='pretrained_detector')]:
        changed=copy.deepcopy(activity); mutate(changed)
        with pytest.raises(ValueError): validate_activity(changed,bundle,tmp_path)


def test_untracked_people_remain_unknown_in_current_frame_conservation():
    row={'participants':[person('head_up')],'detected_people':3}
    add_visible_summary(row)
    assert row['student_state_counts'] == {'head_up':1,'unknown':2}
    assert sum(row['student_state_counts'].values()) == row['detected_people']


def test_frame_evidence_distinguishes_current_action_reference_from_unavailable_temporal_index(tmp_path):
    from test_activity import fixture
    _,activity=fixture(tmp_path)
    frame=activity['frames'][0]
    p=frame['participants'][0]
    p.update(label='read',confidence=.9,source='pretrained_detector',student_state=person('read','pretrained_detector')['student_state'])
    frame['behaviors']={'read':1}
    add_visible_summary(frame)
    assert frame['participation_index'] is None and frame['participation']['index']==80
    text=frame_evidence(frame,1)['text']
    assert 'Behavior participation reference 80.0/100' in text
    assert 'current explicit actions only' in text
    assert 'temporal rule participation index unavailable' in text
