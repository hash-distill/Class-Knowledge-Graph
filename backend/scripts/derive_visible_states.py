"""Re-infer pose on source video; add current states without rewriting raw actions.

This retains the previous boxes, labels, time base, frames and temporal scores.
It does not manufacture tracks for raw detections that were never tracked.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT.parent/'vision/Class_Detection'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(VISION))
from review.common import read_json, atomic_json, sha256
from review.activity import add_visible_summary, frame_evidence, validate_activity, attach_activity, public_metadata


def associate(people, poses, minimum_iou=.35):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not people or not poses: return {}
    affinity=np.zeros((len(people),len(poses)))
    for i,a in enumerate(people):
        for j,b in enumerate(poses):
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            affinity[i,j]=intersection/max(union,1e-8)
    rows,cols=linear_sum_assignment(-affinity)
    return {int(i):(int(j),float(affinity[i,j])) for i,j in zip(rows,cols) if affinity[i,j]>=minimum_iou}


def derive(activity, video, directory, *, imgsz=960, device='cpu', pose_cache=None):
    from src.visible_state import visible_state
    if sha256(video)!=activity['source_sha256']: raise ValueError('Pose video and activity source mismatch')
    import cv2
    result=deepcopy(activity); started=time.perf_counter()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    cached=read_json(pose_cache) if pose_cache else None
    if cached:
        if cached['source_sha256']!=activity['source_sha256'] or len(cached['frames'])!=len(activity['frames']):
            raise ValueError('Pose cache and activity source mismatch')
        pose_meta=cached['model']; raw_frames=cached['frames']
    else:
        import torch
        torch.set_num_threads(8)
        from src.pretrained_pipeline import load_model
        from scripts.fetch_pretrained import load_manifest
        model,pose_meta=load_model('artifacts/models/yolo26n-pose.pt','pose_coco',load_manifest(),device)
        raw_frames=[]
    capture=cv2.VideoCapture(str(video)) if not cached else None
    try:
        for index,row in enumerate(result['frames']):
            if cached:
                raw=raw_frames[index]
                if abs(raw['timestamp_sec']-row['timestamp_sec'])>.001 or raw['frame_id']!=row['id']:
                    raise ValueError('Pose cache timestamp mismatch')
            else:
                capture.set(cv2.CAP_PROP_POS_MSEC,row['timestamp_sec']*1000)
                ok,image=capture.read()
                if not ok or image.shape[:2]!=(row['height'],row['width']): raise ValueError('Source frame decode/dimensions mismatch')
                prediction=model.predict(image,conf=.25,imgsz=imgsz,device=device,verbose=False)[0]
                boxes=prediction.boxes.xyxy.cpu().numpy().tolist() if prediction.boxes is not None else []
                keypoints=prediction.keypoints.data.cpu().numpy().round(4).tolist() if prediction.keypoints is not None else []
                confidence=prediction.boxes.conf.cpu().numpy().tolist() if prediction.boxes is not None else []
                raw={'frame_id':row['id'],'timestamp_sec':row['timestamp_sec'], 'image_url':row['image_url'],
                     'boxes':boxes,'keypoints':keypoints,'confidence':confidence}
                raw_frames.append(raw)
            matches=associate([p['bbox'] for p in row['participants']],raw['boxes'])
            for i,person in enumerate(row['participants']):
                match=matches.get(i); points=raw['keypoints'][match[0]] if match else []
                person['keypoints']=points
                person['pose_association']={'method':'one_to_one_iou','minimum_iou':.35,
                    'status':'matched' if match else 'unmatched','iou':round(match[1],6) if match else None,
                    'source_pose_index':match[0] if match else None}
                person.update(visible_state(points,person['label'],person['confidence'],person['source']))
            add_visible_summary(row)
            if index%10==0 or index==len(result['frames'])-1:
                print(f"Pose states {index+1}/{len(result['frames'])} t={row['timestamp_sec']:.2f} matched={len(matches)} states={row['student_state_counts']}",flush=True)
    finally:
        if capture: capture.release()
    cache={'source_sha256':activity['source_sha256'],'method':'fresh_source_video_pose_prediction',
           'model':public_metadata(pose_meta),'imgsz':imgsz,'device':device,'frames':raw_frames}
    if not cached:
        pose_cache=directory/'activity'/f'pose-v3-{run_id}.json'
        atomic_json(pose_cache,cache)
    result['models']['visible_states']={'method':'coco17_geometry_v1','trained':False,'accuracy_evaluated':False,
        'pose_model':public_metadata(pose_meta),'imgsz':imgsz,'keypoint_confidence_minimum':.5,
        'association':'fresh_source_pose_to_original_track_boxes_one_to_one_iou>=0.35',
        'pose_score_weights':None,'role_inference':False,
        'thresholds':{'head_up_nose_shoulder_gap_ratio_min':.28,'head_lowered_gap_ratio_max':.16,
                      'hand_near_head_wrist_nose_distance_ratio_max':.48,'minimum_shoulder_span_px':12,
                      'standing_min_knee_angle_deg':160,'standing_min_hip_angle_deg':155}}
    result['provenance']['visible_states']={'source_sha256':activity['source_sha256'],
        'source_activity_preserved':True,'pose_cache_file':Path(pose_cache).name,
        'pose_cache_sha256':sha256(pose_cache), 'completed_at':datetime.now(timezone.utc).isoformat(),
        'wall_time_sec':round(time.perf_counter()-started,3),'time_base':'source_media_seconds',
        'original_labels_scores_tracks_preserved':True}
    result['limitations']=list(dict.fromkeys(result['limitations']+[
        '新增头部抬起/降低、手靠近头部、站姿为当前同源帧COCO17关键点几何规则，未经本课人工真值评测；低头无法区分读写与其他活动，抬头不等于注视教师。姿态不单独赋参与分。',
        '已检出参与动作比例按当前明确阅读/书写/举手线索人数除以当前检测人物数；它是可见线索占比，不是全班积极学生比例。未识别不等于未参与。',
        '人物框来自原有检测与镜头内轨迹，重提取姿态仅与原框一一匹配。未自动区分教师/学生或镜中反射，检测人物数不是全班实到人数；未跟踪的人物继续保留待辨。']))
    existing={e['source_refs'][0]:e for e in result['evidence'] if e.get('method')=='visible_behavior_observation' and e.get('source_refs')}
    for row in result['frames']:
        if row['id'] in existing:
            evidence=existing[row['id']]; regenerated=frame_evidence(row,1); regenerated['id']=evidence['id']
            evidence.clear(); evidence.update(regenerated)
    summary=result['summary']
    summary['visible_state_observations']=sum(r['participation']['observed_people'] for r in result['frames'])
    summary['current_action_observations']=sum(r['participation']['active_people'] for r in result['frames'])
    validate_activity(result,{'lesson':{'id':result['frames'][0]['image_url'].split('/')[4],
         'duration_sec':result['duration_sec'],'source_video':{'sha256':result['source_sha256']}}},directory)
    if sha256(video)!=activity['source_sha256']: raise ValueError('Source changed during pose analysis')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--media-dir',type=Path,required=True)
    parser.add_argument('--activity',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--pose-cache',type=Path);parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--device',default='cpu');parser.add_argument('--publish',action='store_true')
    args=parser.parse_args(); path=args.activity or args.media_dir/'activity.json'
    result=derive(read_json(path),args.video,args.media_dir,imgsz=args.imgsz,device=args.device,pose_cache=args.pose_cache)
    result['provenance']['visible_states']['source_activity_sha256']=sha256(path)
    output=args.output or args.media_dir/'activity.visible-v2.json';atomic_json(output,result)
    if args.publish: attach_activity(args.media_dir,result)
    print('Saved',output,flush=True)


if __name__=='__main__': main()
