"""Independent person localization followed by lightweight seven-state crops.

Behavior classification cannot create people. Full-image and tiled COCO models
localize visible people; a small crop classifier retains its own confidence and
abstention status. All outputs remain machine hypotheses, not attention labels.
"""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .dense_detection import (tile_windows, translate_detection, merge_detections,
    exclude_regions, combine_person_sources, upper_body_crop, accept_classification)

SEVEN_CLASSES=['write','read','lookup','turn_head','raise_hand','stand','discuss']


class DenseObserver:
    def __init__(self,weights,training_report,config,device='cuda:0'):
        import torch
        from torchvision.models import mobilenet_v3_small
        from ultralytics import YOLO
        from .pretrained_pipeline import load_model,resolve_project_path
        from scripts.fetch_pretrained import load_manifest
        weights=Path(weights)
        report=json.loads(Path(training_report).read_text(encoding='utf-8'))
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest!=report['sha256'] or report['architecture']!='mobilenet_v3_small' or report['classes']!=SEVEN_CLASSES:
            raise ValueError('Classifier differs from its actual evaluated artifact')
        self.classifier=mobilenet_v3_small(weights=None,num_classes=7)
        self.classifier.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
        self.classifier=self.classifier.to(device).eval()
        person_path=resolve_project_path('artifacts/models/yolo26n.pt')
        person_digest=hashlib.sha256(person_path.read_bytes()).hexdigest()
        if person_digest!=config['person_sha256']:raise ValueError('Person model differs from pinned official asset')
        self.person=YOLO(str(person_path))
        if self.person.task!='detect' or self.person.names.get(0)!='person':raise ValueError('Expected a COCO person-localization model')
        self.pose,pose_meta=load_model('artifacts/models/yolo26n-pose.pt','pose_coco',load_manifest(),device)
        self.config=config;self.device=device;self.half=str(device).startswith('cuda');self._torch=torch
        self.state_names=list(report['classes'])
        self.temperature=report['temperature'];self.threshold=report['confidence_threshold']
        self.mean=torch.tensor(report['normalization']['mean'],device=device).view(1,3,1,1)
        self.std=torch.tensor(report['normalization']['std'],device=device).view(1,3,1,1)
        self.provenance={'pipeline':'dense_person_then_seven_crop_observation_v2',
            'models':[{'file':weights.name,'sha256':digest,'architecture':'mobilenet_v3_small','task':'person_crop_classification',
                       'names':self.state_names,'trained':True,'weight_bytes':weights.stat().st_size,
                       'test_metrics':report['test'],'test_scope':report['scope'],
                       'split_limitations':report.get('split_limitations',[]),
                       'source_image_counts':report['source_image_counts'],'confidence_threshold':self.threshold,
                       'temperature':self.temperature,'target_classroom_accuracy_evaluated':False},
                      {'file':person_path.name,'sha256':person_digest,'task':'person_localization','architecture':'yolo26n.yaml',
                       'weight_bytes':person_path.stat().st_size,'source_url':'https://github.com/ultralytics/assets/releases/tag/v8.4.0'},pose_meta],
            'device':device,'engagement_model':None,'classification':'per_person_upper_body_letterbox_224_rgb',
            'detection':'full_plus_overlapping_tiles_coco_person_and_pose_then_head_aware_dedup',
            'tracking':'all_current_detections_then_scene_local_iou_association',
            'configuration':config,'visible_states':{'method':'trained_crop_classifier_plus_optional_coco17_geometry',
                                                   'role_inference':False,'identity_inference':False}}

    def _predict_people(self,model,image,origin=(0,0),source_size=None,pose=False,imgsz=960):
        conf=self.config.get('pose_conf',.25) if pose else self.config.get('person_conf',.25)
        result=model.predict(image,imgsz=imgsz,conf=conf,iou=.7,max_det=300,classes=None if pose else [0],
                             device=self.device,half=self.half,verbose=False)[0]
        if result.boxes is None:return []
        boxes=result.boxes.xyxy.cpu().numpy().tolist();scores=result.boxes.conf.cpu().numpy().tolist()
        points=result.keypoints.data.cpu().numpy().tolist() if pose and result.keypoints is not None else []
        rows=[]
        for i,(box,score) in enumerate(zip(boxes,scores)):
            row={'bbox':box,'confidence':float(score),'label':'unknown',
                 'keypoints':points[i] if i<len(points) else [],
                 'inference_window':[origin[0],origin[1],origin[0]+image.shape[1],origin[1]+image.shape[0]]}
            width,height=source_size or (image.shape[1],image.shape[0])
            row['source_size']=[width,height]
            rows.append(translate_detection(row,*origin,width,height))
        return rows

    def _classify(self,image,rows):
        if not rows:return rows
        torch=self._torch
        crops=np.stack([upper_body_crop(image,row['bbox'])[:,:,::-1].copy() for row in rows])
        tensor=torch.from_numpy(crops).permute(0,3,1,2).to(self.device,dtype=torch.float32)/255.
        tensor=(tensor-self.mean)/self.std
        outputs=[]
        with torch.inference_mode():
            for chunk in tensor.split(64):
                with torch.autocast('cuda',dtype=torch.float16) if self.half else nullcontext():
                    logits=self.classifier(chunk)
                outputs.append(torch.softmax(logits.float()/self.temperature,1).cpu().numpy())
        for row,probabilities in zip(rows,np.concatenate(outputs)):
            i=int(probabilities.argmax());label=self.state_names[i];confidence=float(probabilities[i])
            accepted=accept_classification(label,confidence,row.get('keypoints',[]),self.threshold)
            row['detection_confidence']=row['confidence']
            row['label']=accepted
            row['confidence']=confidence if accepted!='unknown' else 0.
            row['classification']={'predicted_label':label,'confidence':round(confidence,6),
                'status':'accepted' if accepted!='unknown' else ('below_threshold' if confidence<self.threshold else 'pose_conflict'),
                'probabilities':{name:round(float(value),6) for name,value in zip(self.state_names,probabilities)}}
        return rows

    def observe(self,image):
        started=time.perf_counter();height,width=image.shape[:2];size=self.config.get('full_imgsz',1280)
        people=self._predict_people(self.person,image,imgsz=size)
        poses=self._predict_people(self.pose,image,pose=True,imgsz=size)
        windows=tile_windows(width,height,self.config.get('tile_size',512),self.config.get('overlap',.25))
        if len(windows)>1:
            for x1,y1,x2,y2 in windows:
                crop=image[y1:y2,x1:x2]
                people.extend(self._predict_people(self.person,crop,(x1,y1),(width,height),imgsz=self.config.get('tile_imgsz',640)))
                poses.extend(self._predict_people(self.pose,crop,(x1,y1),(width,height),pose=True,imgsz=self.config.get('tile_imgsz',640)))
        proposals=len(poses)+len(people);poses=merge_detections(poses);people=merge_detections(people)
        before=len(poses)+len(people);excluded=self.config.get('exclude_regions',[])
        poses=exclude_regions(poses,excluded,width,height);people=exclude_regions(people,excluded,width,height)
        excluded_count=before-len(poses)-len(people)
        rows=self._classify(image,combine_person_sources(people,poses))
        if self.half:self._torch.cuda.synchronize()
        return rows,{'method':'full_plus_tiles_person_pose_then_crop_classification','proposal_count':proposals,
                     'retained_people':len(rows),'excluded_detection_proposals':excluded_count,
                     'tile_windows':len(windows) if len(windows)>1 else 0,
                     'wall_seconds':round(time.perf_counter()-started,6),'population_recall_evaluated':False}
