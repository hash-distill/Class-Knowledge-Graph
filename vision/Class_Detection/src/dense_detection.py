"""Source-pixel geometry for dense observations, independent of model libraries.

Current detections are never gated on track confirmation. Local track numbers
only link overlapping boxes within a camera shot; they do not identify people.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def tile_windows(width, height, size=512, overlap=.25):
    if width <= 0 or height <= 0 or size <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid source size or overlap')
    def starts(length):
        if length <= size: return [0]
        step=max(1,int(size*(1-overlap)))
        return sorted(set([*range(0,length-size+1,step),length-size]))
    return [(x,y,min(width,x+size),min(height,y+size))
            for y in starts(height) for x in starts(width)]


def translate_detection(row, x, y, width, height):
    result=deepcopy(row)
    a,b,c,d=row['bbox']
    result['bbox']=[max(0.,a+x),max(0.,b+y),min(float(width),c+x),min(float(height),d+y)]
    for point in result.get('keypoints',[]):
        if point[2]>0:
            point[0]=min(float(width),max(0.,point[0]+x))
            point[1]=min(float(height),max(0.,point[1]+y))
    return result


def overlap(a,b):
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return intersection/max(aa+bb-intersection,1e-8),intersection/max(min(aa,bb),1e-8)


def head_anchor(row):
    points=row.get('keypoints',[])
    if len(points)==17 and points[0][2]>=.35 and points[0][0]>0 and points[0][1]>0:
        return points[0][:2]
    a,b,c,d=row['bbox']
    return [(a+c)/2,b+.18*(d-b)]


def has_head_keypoint(row):
    points=row.get('keypoints',[])
    return len(points)==17 and points[0][2]>=.35 and points[0][0]>0 and points[0][1]>0


def clipped_tile_edges(row):
    window=row.get('inference_window');size=row.get('source_size')
    if not window or not size:return []
    box=row['bbox'];width,height=size
    internal=[window[0]>0,window[1]>0,window[2]<width,window[3]<height]
    # A source-image boundary is not a tile truncation.
    return [i for i in range(4) if internal[i] and abs(box[i]-window[i])<=2.]


def tile_fragment_of(fragment,whole):
    edges=clipped_tile_edges(fragment)
    if not edges:return False
    window=fragment['inference_window'];box=whole['bbox']
    clipped=[max(box[0],window[0]),max(box[1],window[1]),min(box[2],window[2]),min(box[3],window[3])]
    # The larger observation must actually extend beyond an internal crop edge.
    extends=any(box[i]<window[i]-2 if i<2 else box[i]>window[i]+2 for i in edges)
    if not extends:return False
    if overlap(clipped,fragment['bbox'])[0]>=.85:return True
    # A tile can detect just a torso after its head was cut out. This is only
    # suppressed against a complete box with a real head outside that tile.
    if has_head_keypoint(fragment) or not has_head_keypoint(whole):return False
    x,y=head_anchor(whole)
    f=fragment['bbox'];area=(f[2]-f[0])*(f[3]-f[1]);whole_area=(box[2]-box[0])*(box[3]-box[1])
    # Keep the head aligned on the axis the tile did not cut. Otherwise an
    # occluded neighbor below/alongside the complete person could be erased.
    aligned_x=(0 in edges or 2 in edges) and (x<window[0] or x>window[2]) and f[1]-.1*(box[3]-box[1])<=y<=f[3]
    aligned_y=(1 in edges or 3 in edges) and (y<window[1] or y>window[3]) and f[0]-.1*(box[2]-box[0])<=x<=f[2]+.1*(box[2]-box[0])
    return (aligned_x or aligned_y) and area/max(whole_area,1e-8)<=.5 and overlap(box,f)[1]>=.95


def duplicate(a,b,iou_threshold=.5,containment_threshold=.85):
    iou,contained=overlap(a['bbox'],b['bbox'])
    ah,bh=head_anchor(a),head_anchor(b)
    width=min(a['bbox'][2]-a['bbox'][0],b['bbox'][2]-b['bbox'][0])
    # A cut-off torso may widen sideways and narrowly miss a containment cutoff.
    # Near-identical confident noses still locate the same visible head.
    if (has_head_keypoint(a) and has_head_keypoint(b)
            and min(a['keypoints'][0][2],b['keypoints'][0][2])>=.7
            and iou>=.15 and math.dist(ah,bh)<=max(2.,width*.08)):
        return True
    if iou<iou_threshold and contained<containment_threshold:return False
    if math.dist(ah,bh)<=max(12.,width*.4):return True
    # Two confidently different heads can belong to adjacent overlapping people.
    if has_head_keypoint(a) and has_head_keypoint(b):return False
    if iou>=.7 and abs((a['bbox'][0]+a['bbox'][2]-b['bbox'][0]-b['bbox'][2])/2)<=.15*width:
        return True
    return contained>=.95 and (tile_fragment_of(a,b) or tile_fragment_of(b,a))


def merge_detections(rows,iou_threshold=.5,containment_threshold=.85):
    kept=[]
    # Keep the complete observation when a higher-confidence fragment lost its head.
    for row in sorted(rows,key=lambda row:(bool(clipped_tile_edges(row)),-row['confidence'])):
        if not any(duplicate(row,previous,iou_threshold,containment_threshold) for previous in kept):
            kept.append(row)
    return kept


def fuse_detections(poses,behaviors):
    """Behavior detections are independent observations, not pose dependents."""
    assigned={}
    if poses and behaviors:
        affinity=np.zeros((len(behaviors),len(poses)))
        valid=np.zeros_like(affinity,dtype=bool)
        for i,behavior in enumerate(behaviors):
            for j,pose in enumerate(poses):
                iou,inside=overlap(behavior['bbox'],pose['bbox'])
                width=min(behavior['bbox'][2]-behavior['bbox'][0],pose['bbox'][2]-pose['bbox'][0])
                close=math.dist(head_anchor(behavior),head_anchor(pose))<=max(16.,width*.6)
                valid[i,j]=close and (iou>=.2 or inside>=.75)
                affinity[i,j]=.6*iou+.4*inside
        rr,cc=linear_sum_assignment(np.where(valid,-affinity,1e6))
        assigned={int(r):int(c) for r,c in zip(rr,cc) if valid[r,c]}
    rows=[]
    for i,behavior in enumerate(behaviors):
        row=deepcopy(behavior)
        row['detector_sources']=['seven_class_behavior']
        if i in assigned:
            pose=poses[assigned[i]]
            row['keypoints']=deepcopy(pose.get('keypoints',[]))
            row['pose_confidence']=pose['confidence']
            row['detector_sources'].append('coco_pose')
        rows.append(row)
    matched=set(assigned.values())
    for j,pose in enumerate(poses):
        if j not in matched:
            row=deepcopy(pose)
            row['label']='unknown'
            row['detector_sources']=['coco_pose']
            rows.append(row)
    return rows


def combine_person_sources(persons,poses):
    primary=[dict(row,label='unknown') for row in persons]
    rows=fuse_detections(poses,primary)
    for row in rows:
        row['detector_sources']=['coco_person' if name=='seven_class_behavior' else name
                                 for name in row['detector_sources']]
    return merge_detections(rows)


def upper_body_crop(image,box,size=224):
    import cv2
    a,b,c,d=map(float,box)
    height,width=image.shape[:2]
    span=c-a;upper=min(d-b,2.2*span)
    x1=max(0,int(a-.08*span));x2=min(width,int(c+.08*span))
    y1=max(0,int(b-.08*upper));y2=min(height,int(b+upper*1.05))
    crop=image[y1:y2,x1:x2]
    if crop.size==0:raise ValueError('Empty source crop')
    hh,ww=crop.shape[:2];scale=size/max(hh,ww)
    resized=cv2.resize(crop,(max(1,round(ww*scale)),max(1,round(hh*scale))))
    canvas=np.full((size,size,3),114,np.uint8)
    y=(size-resized.shape[0])//2;x=(size-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return canvas


def accept_classification(label,confidence,points,threshold=.6):
    if not math.isfinite(confidence) or confidence<threshold:return 'unknown'
    if label=='raise_hand' and len(points)==17:
        indices=[5,6,9,10]
        if all(points[i][2]>=.5 and points[i][0]>0 and points[i][1]>0 for i in indices):
            span=math.dist(points[5][:2],points[6][:2])
            if span>=12 and all(points[wrist][1]>points[shoulder][1]+.1*span for wrist,shoulder in [(9,5),(10,6)]):
                return 'unknown'
    return label


def exclude_regions(rows,regions,width,height,minimum_inside=.8):
    """Exclude reviewed non-direct regions, not people merely in front of them."""
    excluded=[]
    for region in regions:
        if len(region)!=4 or not all(0<=v<=1 for v in region) or region[0]>=region[2] or region[1]>=region[3]:
            raise ValueError('Exclusion region must be normalized xyxy')
        excluded.append([region[0]*width,region[1]*height,region[2]*width,region[3]*height])
    kept=[]
    for row in rows:
        a=row['bbox'];area=max(1e-8,(a[2]-a[0])*(a[3]-a[1]))
        def inside(b):
            return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))/area
        if not any(inside(b)>=minimum_inside for b in excluded):kept.append(row)
    return kept


class SceneTracker:
    def __init__(self,minimum_iou=.2,max_gap=1.2):
        self.minimum_iou=minimum_iou
        self.max_gap=max_gap
        self._scene=None
        self._next_id=0
        self._tracks={}
        self._last_time=None

    def assign(self,rows,timestamp,scene):
        if not math.isfinite(timestamp) or timestamp<0 or (self._last_time is not None and timestamp<=self._last_time):
            raise ValueError('Track times must increase in source seconds')
        if scene!=self._scene:
            self._tracks={};self._next_id=0;self._scene=scene
        self._tracks={key:value for key,value in self._tracks.items() if timestamp-value['time']<=self.max_gap}
        ids=list(self._tracks)
        matched={}
        if rows and ids:
            affinity=np.array([[overlap(row['bbox'],self._tracks[tid]['bbox'])[0] for tid in ids] for row in rows])
            rr,cc=linear_sum_assignment(-affinity)
            matched={int(r):ids[int(c)] for r,c in zip(rr,cc) if affinity[r,c]>=self.minimum_iou}
        result=[]
        for i,row in enumerate(rows):
            tid=matched.get(i)
            if tid is None:tid=self._next_id;self._next_id+=1
            current=deepcopy(row)
            current['track_id']=tid
            current['track_method']='scene_local_one_to_one_bbox_iou'
            self._tracks[tid]={'bbox':current['bbox'],'time':timestamp}
            result.append(current)
        self._last_time=timestamp
        return result
