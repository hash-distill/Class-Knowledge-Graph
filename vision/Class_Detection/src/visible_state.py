"""Conservative current-frame posture cues from COCO keypoints.

The geometry is an auditable rule, not a trained action or engagement classifier.
Head elevation cannot establish gaze target, listening, reading, or attention.
"""
from __future__ import annotations
import math

NAMES = {'read':'Reading cue', 'write':'Writing cue', 'raise_hand':'Hand-raising cue',
         'head_up':'Head raised', 'head_lowered':'Head lowered',
         'hand_near_head':'Hand near head', 'standing':'Standing cue', 'unknown':'Unknown state'}
METHOD = 'coco17_geometry_v1'


def visible_state(points, action_label='unknown', action_confidence=0., action_source='unavailable'):
    """Keep model actions primary; emit separate supported postural tags.

    Every inferred tag retains its input keypoint indices and measured geometry.
    All coordinates are source-image pixels. Confidence is keypoint/model support,
    not calibrated accuracy. An uncertain pose is explicitly left unknown.
    """
    try:
        pts = [[float(v) for v in row] for row in points]
        valid = len(pts) == 17 and all(len(row) == 3 and all(math.isfinite(v) for v in row) for row in pts)
    except (TypeError, ValueError):
        valid = False
    tags = []
    def tag(label, indices, evidence, metrics):
        return {'label':label, 'name':NAMES[label], 'source':'pose_geometry_rule',
                'method':METHOD, 'confidence':round(min(pts[i][2] for i in indices),4),
                'keypoint_indices':indices, 'evidence':evidence,
                'metrics':{key:round(value,4) for key,value in metrics.items()}}
    def available(indices): return all(pts[i][2] >= .5 and pts[i][0] > 0 and pts[i][1] > 0 for i in indices)
    def distance(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
    def angle(a,b,c):
        u=(a[0]-b[0],a[1]-b[1]);v=(c[0]-b[0],c[1]-b[1])
        denom=math.hypot(*u)*math.hypot(*v)
        return math.degrees(math.acos(max(-1.,min(1.,(u[0]*v[0]+u[1]*v[1])/denom)))) if denom else 0.
    if valid and available([5,6]):
        span = distance(pts[5],pts[6])
        if span >= 12:
            eye = max([1,2], key=lambda i: pts[i][2])
            if available([0,eye]):
                gap = ((pts[5][1]+pts[6][1])/2-pts[0][1])/span
                if gap >= .28:
                    tags.append(tag('head_up',[0,eye,5,6],
                        ['The nose is at least 0.28 shoulder widths above the shoulder midpoint. This describes geometry, not a gaze target.'],
                        {'nose_shoulder_gap_ratio':gap,'shoulder_span_px':span}))
                elif gap <= .16:
                    tags.append(tag('head_lowered',[0,eye,5,6],
                        ['The nose-to-shoulder midpoint height difference is at most 0.16 shoulder widths. This cannot distinguish reading, writing, or other activities.'],
                        {'nose_shoulder_gap_ratio':gap,'shoulder_span_px':span}))
                for wrist, elbow, shoulder in [(9,7,5),(10,8,6)]:
                    if available([wrist]):
                        near = distance(pts[wrist],pts[0])/span
                        if near <= .48:
                            tags.append(tag('hand_near_head',[0,5,6,wrist],
                                ['The wrist is within 0.48 shoulder widths of the nose. No held object or mental state is inferred.'],
                                {'wrist_nose_distance_ratio':near}))
                        if available([elbow]) and pts[wrist][1] < pts[0][1]-.25*span and pts[elbow][1] < pts[shoulder][1] and pts[wrist][1] < pts[shoulder][1]-.8*span:
                            tags.append(tag('raise_hand',[0,5,6,wrist,elbow],
                                ['The wrist is above the nose and shoulder, and the elbow is above the corresponding shoulder. This is a raised-arm posture cue.'],
                                {'wrist_above_shoulder_ratio':(pts[shoulder][1]-pts[wrist][1])/span}))
            lower=[5,6,11,12,13,14,15,16]
            if available(lower):
                knees=[angle(pts[11],pts[13],pts[15]),angle(pts[12],pts[14],pts[16])]
                hips=[angle(pts[5],pts[11],pts[13]),angle(pts[6],pts[12],pts[14])]
                vertical=all(pts[a][1] < pts[b][1] < pts[c][1] for a,b,c in [(11,13,15),(12,14,16)])
                if vertical and min(knees) >= 160 and min(hips) >= 155:
                    tags.append(tag('standing',lower,['Both hips, knees, and ankles are visible and the legs are nearly straight. No speaking activity or role is inferred.'],
                                    {'minimum_knee_angle_deg':min(knees),'minimum_hip_angle_deg':min(hips)}))
    if action_source == 'pretrained_detector' and action_label in ('read','write','raise_hand'):
        primary={'label':action_label,'name':NAMES[action_label],'source':'pretrained_detector',
                 'method':'scb_pretrained_detection','confidence':round(float(action_confidence),4),
                 'evidence':['Original SCB behavior detection. Model labels and confidence are preserved; posture tags are separate.'],
                 'keypoint_indices':[],'metrics':{}}
    else:
        priority={'raise_hand':0,'standing':1,'hand_near_head':2,'head_lowered':3,'head_up':4}
        primary=min(tags,key=lambda row:priority[row['label']]) if tags else {
            'label':'unknown','name':NAMES['unknown'],'source':'unavailable','method':METHOD,
            'confidence':0.,'keypoint_indices':[],'metrics':{},
            'evidence':['Current keypoints are insufficient or geometry is uncertain. The previous frame state is not carried forward.']}
    return {'student_state':primary,'posture_tags':tags}
