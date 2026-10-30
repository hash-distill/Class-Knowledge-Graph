"""Transparent visual-state rule, separate from action evidence and attention.

Weights reuse the project's legacy action policy for recognized model classes.
Turning and standing are intentionally unscored: neither establishes a negative
state or a student's role. Geometry-only head elevation is never scored here.
"""
WEIGHTS={'read':.80,'write':.85,'lookup':.85,'raise_hand':.95,'discuss':.75}
MINIMUM_SUPPORT=.60
MINIMUM_PEOPLE=3
MAXIMUM_GAP=1.2


def build_visual_reference(row,previous=None):
    total=row['detected_people']
    values=[]
    for person in row['participants']:
        state=person.get('student_state',{})
        label=state.get('label')
        if (state.get('source') in ('pretrained_detector','trained_classifier') and person.get('source')==state.get('source')
                and person.get('label')==label and label in WEIGHTS):
            values.append(WEIGHTS[label])
    support=len(values)/total if total else None
    enough=total>=MINIMUM_PEOPLE and support>=MINIMUM_SUPPORT
    index=round(sum(values)/len(values)*100,4) if enough and values else None
    smoothed=index
    old=(previous or {}).get('visual_reference',{})
    if (index is not None and old.get('smoothed_index') is not None
            and previous['scene_id']==row['scene_id']
            and 0<row['timestamp_sec']-previous['timestamp_sec']<=MAXIMUM_GAP):
        smoothed=round(.3*old['smoothed_index']+.7*index,4)
    return {'semantics':'configured_visible_behavior_reference_v1',
            'status':'no_people' if not total else ('estimated' if index is not None else 'insufficient_support'),
            'index':index,'smoothed_index':smoothed,'scored_people':len(values),'total_people':total,
            'support_fraction':round(support,6) if support is not None else None,
            'weights':dict(WEIGHTS),'weight_source':'legacy_project_action_policy_subset',
            'minimum_support_fraction':MINIMUM_SUPPORT,'minimum_people':MINIMUM_PEOPLE,
            'smoothing':{'previous':.3,'current':.7,'maximum_gap_sec':MAXIMUM_GAP},
            'calibrated_engagement_probability':False,
            'population_scope':'directly_detected_people_roles_unverified',
            'definition':'仅对模型识别的读、写、抬头、举手、讨论类别应用原项目规则权重。覆盖不足60%或少于3个对象时不出分；转头、站立、未知与几何抬头均不赋分。0.3旧值+0.7当前值仅在连续同镜头有效观测间平滑；不是心理积极度或知识掌握概率。'}


def validate_visual_reference(row,previous=None):
    expected=build_visual_reference(row,previous)
    actual=row.get('visual_reference')
    if not isinstance(actual,dict) or any(actual.get(key)!=value for key,value in expected.items()):
        raise ValueError('Visual reference does not match its recorded model states and rule')
