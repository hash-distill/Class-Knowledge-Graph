"""Geometry/tracking regressions independent of learned model accuracy."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dense_detection import tile_windows, translate_detection, merge_detections, exclude_regions, SceneTracker, fuse_detections, combine_person_sources, upper_body_crop, accept_classification


def detection(box, confidence=.8, nose=None, label='unknown'):
    points=[[0.,0.,0.] for _ in range(17)] if nose else []
    if nose: points[0]=[*nose,.9]
    return {'bbox':list(box),'confidence':confidence,'keypoints':points,'label':label}


def test_slices_cover_source_edges_without_duplicate_windows():
    windows=tile_windows(1280,720,512,.25)
    assert len(windows)==len(set(windows))==6
    assert (768,208,1280,720) in windows
    assert tile_windows(200,150,512,.25)==[(0,0,200,150)]
    with pytest.raises(ValueError):tile_windows(1280,720,512,1)


def test_crop_coordinates_and_confidence_are_retained_in_source_pixels():
    original=detection([5,10,105,210],nose=(35,30))
    shifted=translate_detection(original,384,208,1280,720)
    assert shifted['bbox']==[389,218,489,418]
    assert shifted['keypoints'][0]==[419,238,.9]
    assert shifted['keypoints'][1]==[0,0,0]
    assert original['bbox']==[5,10,105,210]


def test_contained_crop_duplicate_removed_but_neighbor_head_preserved():
    full=detection([10,10,110,250],.8,nose=(55,40))
    partial=detection([20,15,100,130],.9,nose=(56,41))
    neighbor=detection([20,25,120,230],.85,nose=(100,100))
    merged=merge_detections([full,partial,neighbor])
    assert len(merged)==2
    assert partial in merged and neighbor in merged


def test_real_tile_edge_fragment_uses_window_geometry_instead_of_fallback_head():
    # Reviewed source frame 0: the tile cuts off the real nose at x=760.5.
    full=detection([710.8,287.8,824,415.6],.81,nose=(760.5,321.75))
    full.update(inference_window=[384,208,896,720],source_size=[1280,720])
    fragment=detection([768.1375,288,823.3,414.8],.89)
    fragment.update(inference_window=[768,208,1280,720],source_size=[1280,720])
    assert merge_detections([fragment,full])==[full]


def test_containment_without_tile_clipping_does_not_merge_distinct_people():
    full=detection([710.8,287.8,824,415.6],.81,nose=(760.5,321.75))
    neighbor=detection([768.1375,288,823.3,414.8],.89,nose=(795,311))
    full.update(inference_window=[0,0,1280,720],source_size=[1280,720])
    neighbor.update(inference_window=[768,208,1280,720],source_size=[1280,720])
    assert len(merge_detections([full,neighbor]))==2


def test_nested_box_without_matching_clipped_extent_is_not_a_duplicate():
    full=detection([710,200,824,430],.81,nose=(760,240))
    neighbor=detection([768,288,823,415],.89)
    full.update(inference_window=[0,0,1280,720],source_size=[1280,720])
    neighbor.update(inference_window=[768,208,1280,720],source_size=[1280,720])
    assert len(merge_detections([full,neighbor]))==2


def test_reviewed_headless_torso_fragment_does_not_become_a_second_person():
    full=detection([467.35,344.4,597.4,499.6],.8,nose=(524.6,388.0))
    full.update(inference_window=[384,208,896,720],source_size=[1280,720])
    fragment=detection([468,398,511.6,474],.9)
    fragment.update(inference_window=[0,208,512,720],source_size=[1280,720])
    assert merge_detections([full,fragment])==[full]
    # An independently visible neighboring head must still be retained.
    fragment['keypoints']=detection(fragment['bbox'],nose=(490,411))['keypoints']
    assert len(merge_detections([full,fragment]))==2


def test_high_iou_source_edge_body_without_a_visible_head_is_merged():
    full=detection([0,256.6,62.45,549.2],.8)
    partial=detection([.375,321.6,61,542.4],.7)
    assert len(merge_detections([full,partial]))==1


def test_same_reliable_head_is_not_split_by_containment_rounding_at_tile_bottom():
    full=detection([901.8,432.8,1065.8,642],.9,nose=(974,485.25))
    full.update(inference_window=[768,208,1280,720],source_size=[1280,720])
    partial=detection([884.7,432.8,1079.2,512],.92,nose=(975.2,485.6))
    partial.update(inference_window=[768,0,1280,512],source_size=[1280,720])
    assert merge_detections([partial,full])==[full]


def test_mirror_region_exclusion_does_not_remove_real_person_in_front():
    reflected=detection([820,60,920,150])
    direct=detection([880,140,960,290])
    kept=exclude_regions([reflected,direct],[[.63,.06,.83,.23]],1280,720)
    assert kept==[direct]


def test_behavior_only_person_is_not_dropped_when_pose_model_misses_them():
    row=detection([10,10,80,120],label='lookup')
    assert fuse_detections([], [row])[0]['label']=='lookup'


def test_pose_fusion_keeps_model_behavior_and_does_not_duplicate_person():
    pose=detection([10,10,110,240],.95,nose=(55,40))
    action=detection([20,15,100,130],.7,label='read')
    rows=fuse_detections([pose],[action])
    assert len(rows)==1 and rows[0]['label']=='read'
    assert rows[0]['confidence']==.7
    assert rows[0]['keypoints'][0]==[55,40,.9]


def test_current_boxes_are_emitted_immediately_and_missing_boxes_not_invented():
    tracker=SceneTracker()
    first=tracker.assign([detection([10,10,100,200])],0,1)
    assert first[0]['track_id']==0
    second=tracker.assign([detection([12,12,102,202]),detection([200,10,300,200])],.5,1)
    assert [r['track_id'] for r in second]==[0,1]
    assert tracker.assign([] ,1,1)==[]
    after_gap=tracker.assign([detection([12,12,102,202])],3,1)
    assert after_gap[0]['track_id']!=0


def test_scene_cut_and_one_to_one_assignment_never_reuse_one_track_twice():
    tracker=SceneTracker()
    tracker.assign([detection([10,10,100,200])],0,1)
    result=tracker.assign([detection([12,12,102,202]),detection([15,15,105,205])],.5,1)
    assert len({r['track_id'] for r in result})==2
    cut=tracker.assign([detection([12,12,102,202])],1,2)
    assert cut[0]['track_id']==0  # Identifier is scoped to scene 2, not the previous person.


def test_separate_person_detector_can_localize_without_pose_or_behavior_confirmation():
    people=combine_person_sources([detection([20,20,100,180])],[])
    assert len(people)==1 and people[0]['detector_sources']==['coco_person']
    assert combine_person_sources([],[])==[]  # A classifier cannot invent a person on a slide.


def test_crop_uses_same_letterbox_geometry_as_annotated_training_crops():
    import numpy as np
    image=np.full((200,300,3),220,np.uint8)
    crop=upper_body_crop(image,[20,10,100,195])
    assert crop.shape==(224,224,3)
    assert (crop[:,0]==114).all()
    assert (crop[112,112]==220).all()


def test_low_confidence_and_confidently_contradicted_hand_raise_are_left_unknown():
    assert accept_classification('lookup',.4,[],.6)=='unknown'
    assert accept_classification('lookup',.8,[],.6)=='lookup'
    points=[[0,0,0] for _ in range(17)]
    for i,p in {5:[20,40,.9],6:[60,40,.9],9:[20,90,.9],10:[60,90,.9]}.items():points[i]=p
    assert accept_classification('raise_hand',.95,points,.6)=='unknown'
    assert accept_classification('raise_hand',.95,[],.6)=='raise_hand'  # Unseen wrists are unknown, not contradictory.
