from copy import deepcopy
import pytest
from review.visual_reference import build_visual_reference, validate_visual_reference


def frame(labels, timestamp=0, scene=1):
    return {'detected_people':len(labels),'timestamp_sec':timestamp,'scene_id':scene,
            'participants':[{'label':label,'source':'pretrained_detector' if label!='head_up' else 'pose_rule',
                             'student_state':{'label':label,'source':'pretrained_detector' if label!='head_up' else 'pose_geometry_rule'}} for label in labels]}


def test_trained_lookup_is_a_named_legacy_rule_not_geometry_attention():
    row=frame(['lookup','lookup','lookup'])
    ref=build_visual_reference(row)
    assert ref['index']==85 and ref['support_fraction']==1
    assert ref['semantics']=='configured_visible_behavior_reference_v1'
    assert ref['calibrated_engagement_probability'] is False
    assert build_visual_reference(frame(['head_up']*3))['index'] is None
    for person in row['participants']:
        person['source']=person['student_state']['source']='trained_classifier'
    assert build_visual_reference(row)['index']==85


def test_sparse_known_states_and_unknown_are_not_a_valid_zero_score():
    ref=build_visual_reference(frame(['read','unknown','unknown','unknown']))
    assert ref['index'] is None and ref['smoothed_index'] is None
    assert ref['status']=='insufficient_support' and ref['scored_people']==1
    assert ref['support_fraction']==.25
    assert build_visual_reference(frame([]))['index'] is None


def test_turning_or_standing_is_visible_but_does_not_imply_disengagement_or_role():
    ref=build_visual_reference(frame(['turn_head','stand','unknown']))
    assert ref['scored_people']==0 and ref['index'] is None
    assert 'turn_head' not in ref['weights'] and 'stand' not in ref['weights']


def test_ppt_smoothing_only_combines_contiguous_observed_source_samples():
    old=frame(['read']*3);old['visual_reference']=build_visual_reference(old)
    now=frame(['lookup']*3,.5)
    assert build_visual_reference(now,old)['smoothed_index']==83.5
    assert build_visual_reference(frame(['lookup']*3,.5,2),old)['smoothed_index']==85
    assert build_visual_reference(frame(['lookup']*3,3),old)['smoothed_index']==85
    gap=frame(['unknown']*3,.5);gap['visual_reference']=build_visual_reference(gap,old)
    assert build_visual_reference(frame(['lookup']*3,1),gap)['smoothed_index']==85


def test_reference_tampering_is_rejected_and_source_rows_are_unchanged():
    row=frame(['read']*3);saved=deepcopy(row)
    row['visual_reference']=build_visual_reference(row)
    validate_visual_reference(row,None)
    assert {k:v for k,v in row.items() if k!='visual_reference'}==saved
    row['visual_reference']['index']=99
    with pytest.raises(ValueError,match='reference'):validate_visual_reference(row,None)
