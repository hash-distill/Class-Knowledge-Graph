"""Re-infer a registered source video with evaluated lightweight seven-state weights.

Writes a new artifact first. Publishing is explicit and preserves prior outputs.
This is an offline model worker, not an API-process import.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
VISION=ROOT.parent/'vision/Class_Detection'
sys.path.insert(0,str(VISION));sys.path.insert(0,str(ROOT))
from review.common import read_json,atomic_json,sha256
from review.activity import validate_activity,add_visible_summary,frame_evidence,attach_activity
from review.visual_reference import build_visual_reference


def analyze(args):
    import cv2
    import torch
    from src.dense_inference import DenseObserver
    from src.dense_detection import SceneTracker
    from src.visible_state import visible_state
    from src.observation import TemporalEvidence,CONTEXT_SCORES,keypoint_quality
    registry=read_json(ROOT/'configs/lessons.json')
    entry=registry[args.lesson]
    directory=Path(entry['directory']);video=Path(entry['source_video'])
    original=read_json(directory/'activity.json')
    bundle=read_json(directory/'bundle.json')
    config=read_json(args.config)
    if sha256(video)!=original['source_sha256'] or config['source_sha256']!=original['source_sha256']:
        raise ValueError('Dense configuration, activity and source video must match')
    torch.set_num_threads(6)
    if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    observer=DenseObserver(args.weights,args.training_report,config,args.device)
    tracker=SceneTracker()
    temporal=TemporalEvidence()
    result=deepcopy(original)
    started=time.perf_counter();last_scene=None;previous=None
    capture=cv2.VideoCapture(str(video))
    try:
        for index,row in enumerate(result['frames']):
            capture.set(cv2.CAP_PROP_POS_MSEC,row['timestamp_sec']*1000)
            ok,image=capture.read()
            if not ok or image.shape[:2]!=(row['height'],row['width']):raise ValueError('Source decode failed')
            if row['scene_id']!=last_scene:temporal.tracks.clear();last_scene=row['scene_id']
            detections,review=observer.observe(image)
            review['previous_detected_people']=row['detected_people']
            people=[]
            for detection in tracker.assign(detections,row['timestamp_sec'],row['scene_id']):
                label=detection['label'];points=detection.get('keypoints',[])
                trained=label!='unknown'
                confidence=detection['confidence'] if trained else 0.
                source='trained_classifier' if trained else 'pose_rule'
                scores=CONTEXT_SCORES['unspecified']
                score=scores.get(label) if trained else None
                temporal.push(detection['track_id'],row['timestamp_sec'],label,confidence,score,'unspecified')
                cas,assessment=temporal.assess(detection['track_id'],row['timestamp_sec'],'unspecified')
                person={'track_id':f"S{row['scene_id']}-T{detection['track_id']}",
                        'bbox':[round(v,4) for v in detection['bbox']], 'label':label,'confidence':round(confidence,6),
                        'score':cas if score is not None and assessment.status=='estimated' else None,
                        'source':source,'assessment':assessment.model_dump(mode='json'),
                        'keypoints':[[round(v,4) for v in p] for p in points],
                        'detector_sources':detection['detector_sources'],
                        'detection_confidence':round(detection.get('detection_confidence',detection['confidence']),6),
                        'classification':detection['classification'],
                        'inference_window':detection.get('inference_window'),
                        'source_size':detection.get('source_size'),
                        'track_method':detection['track_method']}
                pose=visible_state(points,'unknown',0.,'pose_geometry_rule')
                if trained:
                    names={'write':'Writing cue','read':'Reading cue','lookup':'Head-up cue','turn_head':'Head-turning cue',
                           'raise_hand':'Hand-raising cue','stand':'Standing cue','discuss':'Discussion cue'}
                    pose['student_state']={'label':label,'name':names[label],'source':'trained_classifier',
                        'method':'scbehavior_seven_mobilenetv3_small','confidence':round(confidence,6),
                        'keypoint_indices':[],'metrics':{},
                        'evidence':[f'Seven-class MobileNetV3 Small prediction for the current detected-object crop: {label}; class confidence is not accuracy or a probability of psychological engagement']}
                person.update(pose)
                people.append(person)
            temporal.prune(row['timestamp_sec'])
            counts=Counter(p['label'] for p in people)
            scored=[p['score'] for p in people if p['score'] is not None]
            row.update(participants=people,detected_people=len(people),known_people=len(scored),
                behaviors=dict(counts),coverage=round(len(scored)/len(people),6) if people else 0,
                participation_index=round(sum(scored)/len(scored)*100,4) if scored else None,
                status='no_people' if not people else ('observed' if scored else 'low_evidence'),
                detection_review=review)
            add_visible_summary(row)
            row['visual_reference']=build_visual_reference(row,previous)
            previous=row
            if index%10==0 or index==len(result['frames'])-1:
                print(f"Dense {index+1}/{len(result['frames'])} t={row['timestamp_sec']:.2f} boxes={len(people)} states={row['student_state_counts']} reference={row['visual_reference']['index']}",flush=True)
    finally:capture.release()
    result['models']['observation']=observer.provenance
    result['models']['runtime'].update(device=args.device,imgsz=config['full_imgsz'],python=sys.version.split()[0])
    result['models']['visible_states']={'method':'seven_class_mobilenetv3_small_plus_pose_geometry','trained_state_classes':observer.state_names,
        'accuracy_evaluated':True,'accuracy_scope':'classification given annotated crops from SCBehavior fixed test images; classroom/scene overlap exists across splits; not independent classroom validation','target_classroom_accuracy_evaluated':False}
    result['provenance']['dense_revision']={'started_from_activity_sha256':sha256(directory/'activity.json'),
        'source_sha256':result['source_sha256'],'time_base':'source_media_seconds','completed_at':datetime.now(timezone.utc).isoformat(),
        'wall_seconds':round(time.perf_counter()-started,3),
        'peak_allocated_gpu_mib':round(torch.cuda.max_memory_allocated()/2**20,2) if str(args.device).startswith('cuda') else None,
        'config_sha256':sha256(args.config),'training_report_sha256':sha256(args.training_report),
        'old_activity_preserved':True,'scene_boundaries':'preserved existing sampled-source cuts; not re-estimated'}
    result['summary'].update(person_observations=sum(r['detected_people'] for r in result['frames']),
        known_observations=sum(r['known_people'] for r in result['frames']),
        visible_state_observations=sum(r['participation']['observed_people'] for r in result['frames']),
        current_action_observations=sum(r['participation']['active_people'] for r in result['frames']))
    result['limitations']=[text for text in result['limitations'] if 'Person boxes come from existing detections' not in text and 'No trained context model is configured' not in text and 'Pretrained reading/writing/hand-raising detection' not in text]
    result['limitations']+=['The added seven-class classifier was trained on a local SCBehavior subset and evaluated on annotated crops from fixed test images. Training and test sets include nearby frames from the same lessons. This result is not independent lesson validation, end-to-end detection accuracy, current-lesson accuracy, or role-recognition validation.',
        'Person localization uses independent COCO person and pose models on full images and overlapping tiles, with global deduplication and one-to-one fusion. The behavior classifier generates no new person boxes. The reflection exclusion region is a reviewed rear-wall reflection area for this segment and does not automatically transfer to other videos.',
        'The visible-behavior rule reference reuses the original project weights for reading, writing, head-up, hand raising, and discussion. Head turning, standing, unknown states, and geometric head-up cues receive no score. No score is produced with support below 60% or fewer than 3 objects. It is not a probability of psychological engagement.',
        'Smoothing uses 0.3 previous + 0.7 current only between consecutive valid observations in the same shot; missing observations and shot changes reset it. Detected-object counts do not represent class size, and state support is not detection recall.']
    old_evidence={e['source_refs'][0]:e['id'] for e in result['evidence'] if e.get('method')=='visible_behavior_observation' and e.get('source_refs')}
    other=[e for e in result['evidence'] if e.get('method')!='visible_behavior_observation']
    evidence=[]
    for index,row in enumerate(result['frames']):
        record=frame_evidence(row,index+1)
        if row['id'] in old_evidence:record['id']=old_evidence[row['id']]
        evidence.append(record)
    result['evidence']=evidence+other
    validate_activity(result,bundle,directory)
    if sha256(video)!=original['source_sha256']:raise ValueError('Video changed during dense inference')
    atomic_json(args.output,result)
    if args.publish:attach_activity(directory,result)
    print('Saved',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson',default='unified-chemistry-001-9')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--training-report',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--publish',action='store_true')
    analyze(parser.parse_args())


if __name__=='__main__':main()
