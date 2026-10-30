"""Validated same-source behavior overlays; model imports are inference-only and lazy."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import math
import re

from .common import atomic_json, read_json, sha256, valid_id
from .mutation import LessonMutationLock

REPORT_TITLE = '可见行为与课堂情境'
NO_BEHAVIOR_LIMIT = '未提供与本课确认同步的学生行为录像，未计算投入度、知识掌握或教学效果。'
# Historical generated labels are data identifiers for updating older bundles.
# Keep them as aliases so a language change cannot duplicate stale report rows.
REPORT_TITLES = {REPORT_TITLE, 'Visible behavior and lesson context'}
NO_BEHAVIOR_LIMITS = {NO_BEHAVIOR_LIMIT, 'No student behavior video has been confirmed as synchronized with this lesson; engagement, knowledge mastery, and teaching effectiveness were not calculated.'}
VISIBLE_ACTION_WEIGHTS = {'read': .80, 'write': .85, 'raise_hand': .95}
VISIBLE_STATE_LABELS = {'read','write','raise_hand','head_up','head_lowered','hand_near_head','standing','unknown',
                        'lookup','turn_head','stand','discuss'}


def visible_participation(row):
    """Current-frame action evidence; unknown/posture never receive a low score."""
    total = row['detected_people']
    states = [p.get('student_state', {}).get('label','unknown') for p in row['participants']]
    observed = sum(label != 'unknown' for label in states)
    actions = [VISIBLE_ACTION_WEIGHTS[label] for label in states if label in VISIBLE_ACTION_WEIGHTS]
    return {'semantics':'visible_participation_rule_v2', 'population_scope':'visible_people_roles_unverified',
            'status':'no_people' if not total else ('observed' if observed else 'low_evidence'),
            'index':round(sum(actions)/len(actions)*100,4) if actions else None,
            'total_people':total, 'observed_people':observed, 'unknown_people':total-observed,
            'coverage':round(observed/total,6) if total else None,
            'active_people':len(actions), 'active_fraction':round(len(actions)/total,6) if total else None,
            'scored_people':len(actions), 'score_coverage':round(len(actions)/total,6) if total else None,
            'weights':dict(VISIBLE_ACTION_WEIGHTS),
            'definition':'已检出参与动作比例=当前阅读/书写/举手线索人数÷当前检测人物数。未识别不等于未参与；姿态单独展示且不赋参与分。参考分只在明确动作内按公开权重求均值。'}


def add_visible_summary(row):
    states = Counter(p.get('student_state',{}).get('label','unknown') for p in row['participants'])
    states['unknown'] += row['detected_people']-len(row['participants'])
    row['student_state_counts'] = {key:value for key,value in states.items() if value}
    row['participation'] = visible_participation(row)
    return row


def number(value, low=0, high=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Activity contains invalid numeric values')
    return value


def count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Activity counts must be nonnegative integers')
    return value


def interval(row, duration):
    start, end = number(row.get('start_sec')), number(row.get('end_sec'))
    if not 0 <= start < end <= duration + .001:
        raise ValueError('Activity time outside source duration')
    return start, end


def validate_activity(activity, bundle, media_dir=None):
    """Return the original artifact only after source, time and evidence checks pass."""
    lesson = bundle['lesson']; duration = number(lesson['duration_sec'], .000001)
    if activity.get('schema_version') != '1.0' or activity.get('status') != 'complete':
        raise ValueError('Activity analysis is incomplete or unsupported')
    digest = activity.get('source_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or digest != lesson['source_video']['sha256']:
        raise ValueError('Activity and lesson belong to different sources')
    if 'duration_sec' in activity and abs(number(activity['duration_sec']) - duration) > .01:
        raise ValueError('Activity source duration mismatch')
    sampling = number(activity.get('sampling_interval_sec'), .000001)
    if not isinstance(activity.get('models'), dict) or not activity['models'] or not isinstance(activity.get('limitations'), list):
        raise ValueError('Activity requires model provenance and limitations')
    reserved = {r['id'] for key in ('transcript', 'evidence_blocks', 'slides', 'courseware_evidence', 'revised_evidence_blocks')
                for r in bundle.get(key, [])}
    identifiers = set(reserved)
    def identify(row):
        identifier = row.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,60}', identifier) or identifier in identifiers:
            raise ValueError('Activity IDs must be globally unique')
        identifiers.add(identifier)
    contexts = activity.get('context_segments', [])
    for row in contexts:
        identify(row); interval(row, duration)
        if row.get('method') != 'trained_context' or not row.get('model_id') or not isinstance(row.get('text'), str):
            raise ValueError('Context must identify an actual trained classifier')
        if not isinstance(row.get('labels'), list):
            raise ValueError('Context labels missing')
        for label in row['labels']:
            number(label.get('probability'), 0, 1)
            if not label.get('label') or not label.get('name'): raise ValueError('Invalid context labels')
    frames = activity.get('frames')
    if not isinstance(frames, list) or not frames: raise ValueError('Activity requires observed frames')
    previous = None; previous_scene = 1
    prefix = f"/api/review/media/{lesson['id']}/activity/frames/"
    for row in frames:
        identify(row); start, end = interval(row, duration)
        timestamp = number(row.get('timestamp_sec'))
        if abs(start - timestamp) > .001 or not start <= timestamp < end:
            raise ValueError('Frame timestamp must start its observed sample interval')
        if previous is None:
            if start > .01 or row.get('scene_id') != 1 or row.get('scene_cut') is not False:
                raise ValueError('Activity must begin at source time zero in scene one')
        else:
            if timestamp <= previous['timestamp_sec'] or abs(previous['end_sec'] - start) > .001 or start-previous['timestamp_sec'] > sampling+.01:
                raise ValueError('Activity sample gaps or overlapping frame intervals')
            if not isinstance(row.get('scene_cut'), bool) or row.get('scene_id') != previous_scene + int(row['scene_cut']):
                raise ValueError('Activity scene changes must reset tracking')
        previous_scene = count(row.get('scene_id'))
        url = row.get('image_url', '')
        if not isinstance(url, str): raise ValueError('Activity image URL missing')
        parsed = urlsplit(url); decoded = unquote(parsed.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or not decoded.startswith(prefix) or '..' in decoded.split('/') or '\\' in decoded or ':' in decoded:
            raise ValueError('Activity image must be inside the same lesson media directory')
        if Path(decoded).suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            raise ValueError('Unsupported activity image type')
        if media_dir is not None:
            root = Path(media_dir).resolve(); image = (root / decoded.split(f"/api/review/media/{lesson['id']}/", 1)[1]).resolve()
            if not image.is_relative_to(root) or not image.is_file(): raise ValueError('Activity image missing or outside lesson')
        detected, known = count(row.get('detected_people')), count(row.get('known_people'))
        participants = row.get('participants')
        if not isinstance(participants, list) or not known <= len(participants) <= detected:
            raise ValueError('Activity participant counts mismatch')
        scored = []; tracks = set(); behavior_counts = Counter()
        for person in participants:
            track = person.get('track_id', '')
            if not isinstance(track, str) or not re.fullmatch(fr'S{previous_scene}-T\d+', track) or track in tracks:
                raise ValueError('Activity track must be unique within its scene')
            tracks.add(track)
            box = person.get('bbox')
            if not isinstance(box, list) or len(box) != 4: raise ValueError('Invalid participant bbox')
            for value in box: number(value)
            if box[0] >= box[2] or box[1] >= box[3]: raise ValueError('Invalid participant bbox')
            if 'width' in row and box[2] > number(row['width'], 1)+1: raise ValueError('BBox outside source pixels')
            if 'height' in row and box[3] > number(row['height'], 1)+1: raise ValueError('BBox outside source pixels')
            number(person.get('confidence'), 0, 1)
            if not isinstance(person.get('label'), str) or not person.get('source'): raise ValueError('Missing observation source')
            behavior_counts[person['label']] += 1
            if person.get('score') is not None:
                if person['label'] == 'unknown': raise ValueError('Unknown behavior cannot carry a participation score')
                scored.append(number(person['score'], 0, 1))
        behavior_counts['unknown'] += detected-len(participants)
        behavior_counts = {key: value for key, value in behavior_counts.items() if value}
        if row.get('behaviors') != behavior_counts or known != len(scored): raise ValueError('Activity behavior or known counts mismatch')
        coverage = known/detected if detected else 0
        if abs(number(row.get('coverage'), 0, 1) - coverage) > .0001: raise ValueError('Activity coverage mismatch')
        expected_status = 'no_people' if not detected else ('observed' if scored else 'low_evidence')
        if row.get('status') != expected_status: raise ValueError('Activity observation status mismatch')
        expected_index = sum(scored)/len(scored)*100 if scored else None
        if expected_index is None:
            if row.get('participation_index') is not None: raise ValueError('Unavailable participation must remain null')
        elif abs(number(row.get('participation_index'), 0, 100)-expected_index) > .01:
            raise ValueError('Activity rule index mismatch')
        if end-start > sampling+.01: raise ValueError('Frame interval exceeds sample spacing')
        if 'participation' in row or 'student_state_counts' in row:
            expected_counts = Counter()
            for person in participants:
                state = person.get('student_state', {})
                if state.get('label') not in VISIBLE_STATE_LABELS or not state.get('name') or not state.get('evidence'):
                    raise ValueError('Visible state requires a supported label and evidence')
                number(state.get('confidence'),0,1)
                if state.get('source') not in ('pretrained_detector','trained_classifier','pose_geometry_rule','unavailable'):
                    raise ValueError('Visible state source missing')
                if state['source'] in ('pretrained_detector','trained_classifier') and (state['label'] != person['label'] or person['source'] != state['source']):
                    raise ValueError('Visible model state must preserve original model observation')
                if state['label'] in ('lookup','turn_head','stand','discuss') and state['source'] not in ('pretrained_detector','trained_classifier'):
                    raise ValueError('Seven-state model labels cannot be invented by geometry rules')
                points = person.get('keypoints', [])
                if not isinstance(points,list) or len(points) not in (0,17): raise ValueError('Visible state requires COCO17 keypoints')
                for point in points:
                    if not isinstance(point,list) or len(point)!=3: raise ValueError('Invalid visible state keypoint')
                    number(point[0]); number(point[1]); number(point[2],0,1)
                tags = person.get('posture_tags',[])
                if not isinstance(tags,list) or any(not isinstance(tag,dict) or tag.get('source')!='pose_geometry_rule' for tag in tags):
                    raise ValueError('Posture tags must identify geometry evidence')
                for tag in [state,*tags]:
                    if tag.get('label') not in VISIBLE_STATE_LABELS or not isinstance(tag.get('keypoint_indices'),list):
                        raise ValueError('Visible state keypoint evidence missing')
                    number(tag.get('confidence'),0,1)
                    if not isinstance(tag.get('evidence'),list) or not tag['evidence']:
                        raise ValueError('Visible state requires explicit evidence')
                    if tag.get('source') == 'pose_geometry_rule':
                        indices = tag['keypoint_indices']
                        if not points or not indices or any(type(i) is not int or not 0<=i<17 or points[i][2]<.5 for i in indices):
                            raise ValueError('Pose state is not supported by confident keypoints')
                expected_counts[state['label']] += 1
            expected_counts['unknown'] += detected-len(participants)
            if row.get('student_state_counts') != {key:value for key,value in expected_counts.items() if value}:
                raise ValueError('Current visible state counts mismatch')
            expected_participation = visible_participation(row)
            if any(row['participation'].get(key) != value for key,value in expected_participation.items()):
                raise ValueError('Current visible participation summary mismatch')
        if 'visual_reference' in row:
            from .visual_reference import validate_visual_reference
            validate_visual_reference(row, previous)
        previous = row
    if abs(frames[-1]['end_sec']-duration) > .01: raise ValueError('Activity does not cover the complete source')
    expected = {'sampled_frames': len(frames), 'scene_count': previous_scene,
                'person_observations': sum(r['detected_people'] for r in frames),
                'known_observations': sum(r['known_people'] for r in frames)}
    summary = activity.get('summary', {})
    if any(count(summary.get(key)) != value for key, value in expected.items()): raise ValueError('Activity summary counts mismatch')
    observations = {r['id']: r for r in [*frames, *contexts]}
    evidence = activity.get('evidence')
    if not isinstance(evidence, list) or not evidence: raise ValueError('Activity requires traceable behavior evidence')
    for row in evidence:
        identify(row); start, end = interval(row, duration)
        if row.get('kind') != 'behavior' or not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Invalid behavior evidence')
        refs = row.get('source_refs')
        sources = [observations.get(ref) for ref in refs] if refs else list(observations.values())
        if not sources or any(source is None for source in sources) or not any(source['start_sec'] <= start and end <= source['end_sec'] for source in sources):
            raise ValueError('Behavior evidence is not supported by an observed interval')
    return activity


def overlay_activity(bundle, activity, media_dir=None):
    validate_activity(activity, bundle, media_dir)
    result = deepcopy(bundle); result['activity_analysis'] = deepcopy(activity)
    lesson = result['lesson']; lesson.setdefault('modalities', {})['behavior'] = 'available'
    lesson.setdefault('provenance', {})['activity'] = {'source_sha256': activity['source_sha256'],
        'schema_version': '1.0', 'time_base': 'source_media_seconds', 'score_semantics': 'uncalibrated_behavior_rule',
        'context_semantics': 'trained_clip_level_prediction'}
    report = result['report']; summary = activity['summary']
    report['sections'] = [r for r in report.get('sections', []) if r.get('title') not in REPORT_TITLES]
    report['sections'].append({'title': REPORT_TITLE,
        'text': f"同源录像抽样 {summary['sampled_frames']} 帧、{summary['scene_count']} 个检测镜头段；累计 {summary['person_observations']} 人次观测，{summary['known_observations']} 人次具有规则参与线索。人数是逐帧观测累计，不是唯一人数。情境为片段级模型预测。",
        'evidence_ids': [r['id'] for r in activity.get('evidence', [])]})
    report['limitations'] = list(dict.fromkeys([r for r in report.get('limitations', []) if r not in NO_BEHAVIOR_LIMITS] + activity['limitations']))
    report.setdefault('statistics', {}).update({'activity_'+key: value for key, value in summary.items()})
    return result


def attach_activity(media_dir, activity):
    """Update reports, base and active overlays under one lease; publish bundle last."""
    from .courseware_service import overlay_bundle, persist_bundle
    directory = Path(media_dir)
    with LessonMutationLock(directory):
        current = read_json(directory/'bundle.json')
        base = read_json(directory/'bundle.base.json') if current.get('courseware') else current
        base = overlay_activity(base, activity, directory)
        result = overlay_bundle(base, current['courseware']) if current.get('courseware') else base
        if current.get('courseware'): atomic_json(directory/'bundle.base.json', base)
        atomic_json(directory/'activity.json', activity)
        persist_bundle(directory, result)
        return result


def snapshot_frame(snapshot, *, index, timestamp, end, scene_id, scene_cut, image_url, width, height):
    participants = []; behaviors = Counter()
    for person in snapshot.get('student_states', []):
        action = person['action']; label = action['label']
        score = person.get('cas') if label != 'unknown' and person.get('assessment', {}).get('status') == 'estimated' else None
        source = 'pretrained_detector' if action.get('source') == 'detection' else 'pose_rule'
        participants.append({'track_id': f"S{scene_id}-T{person['track_id']}", 'bbox': person['bbox'],
                             'label': label, 'confidence': action['confidence'], 'score': score, 'source': source,
                             'assessment': person.get('assessment')})
        if person.get('student_state'):
            participants[-1].update(student_state=person['student_state'], posture_tags=person.get('posture_tags',[]),
                                    keypoints=person.get('keypoints',[]))
        behaviors[label] += 1
    untracked = snapshot['classroom_metrics'].get('untracked_people', 0)
    if untracked: behaviors['unknown'] += untracked
    scores = [r['score'] for r in participants if r['score'] is not None]
    detected = len(participants)+untracked
    row = {'id': f'V{index:04d}', 'start_sec': timestamp, 'end_sec': end, 'timestamp_sec': timestamp,
            'scene_id': scene_id, 'scene_cut': scene_cut, 'image_url': image_url, 'width': width, 'height': height,
            'status': 'no_people' if not detected else ('observed' if scores else 'low_evidence'),
            'detected_people': detected, 'known_people': len(scores), 'coverage': round(len(scores)/detected, 6) if detected else 0,
            'participation_index': round(sum(scores)/len(scores)*100, 4) if scores else None,
            'behaviors': dict(behaviors), 'participants': participants}
    if snapshot.get('provenance',{}).get('visible_states') or any(p.get('student_state') for p in participants):
        add_visible_summary(row)
    return row


def reset_tracking(pipeline):
    """Cut identities and temporal support at a detected scene boundary."""
    for tracker in getattr(getattr(pipeline.pose, 'predictor', None), 'trackers', []): tracker.reset()
    pipeline.evidence.tracks.clear()
    pipeline._last_timestamp = None
    pipeline._track_epoch_offset = 0
    pipeline._max_emitted_track = -1


def scene_difference(previous, current):
    """Normalized downsampled pixel change plus histogram distance; no scene truth labels."""
    import cv2
    import numpy as np
    left = cv2.resize(previous, (96, 54)); right = cv2.resize(current, (96, 54))
    pixel = float(np.abs(left.astype(np.float32)-right.astype(np.float32)).mean()/255)
    def histogram(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    histogram_distance = float(cv2.compareHist(histogram(left), histogram(right), cv2.HISTCMP_BHATTACHARYYA))
    return round(.7*pixel+.3*histogram_distance, 6)


def load_pipeline(config_path, device, imgsz):
    import sys
    visual_root = Path(__file__).resolve().parents[2]/'vision/Class_Detection'
    sys.path.insert(0, str(visual_root))
    from src.pretrained_pipeline import PretrainedClassroomPipeline
    pipeline = PretrainedClassroomPipeline(config_path, device=device)
    pipeline.imgsz = imgsz
    return pipeline


def public_metadata(value):
    """Keep model metrics useful without exposing local source and checkpoint paths."""
    if isinstance(value, dict):
        return {str(key): public_metadata(item) for key, item in value.items()
                if not any(word in str(key).lower() for word in ('path', 'directory', 'root'))}
    if isinstance(value, list): return [public_metadata(item) for item in value]
    if isinstance(value, str) and (re.match(r'^[A-Za-z]:[\\/]', value) or value.startswith(('\\\\', '/mnt/', '/home/', '/Users/'))):
        return Path(value).name
    return value


def load_context(video_path, model_dir, device):
    from learning.ts_context import predict_context
    directory = Path(model_dir)
    contexts = predict_context(video_path, directory, device=device)
    model = read_json(directory/'model.json')
    metrics = read_json(directory/'metrics.json')
    return contexts, {'method': 'trained_context', 'scope': 'clip_level_prediction',
                      'model': public_metadata({key: value for key, value in model.items()
                          if key in ('model_id', 'architecture', 'encoder', 'encoder_sha256', 'encoder_revision',
                                     'frames_per_clip', 'feature_dimension', 'training_count', 'dataset_count', 'labels', 'method')}),
                      'metrics': compact_metrics(metrics), 'metrics_sha256': sha256(directory/'metrics.json')}


def compact_metrics(report):
    """Export scalar training/evaluation facts; retain full detailed files offline."""
    def scalars(record):
        return {key: value for key, value in record.items()
                if value is None or isinstance(value, (str, int, float, bool))}
    result = scalars(report)
    for key in ('split_counts', 'validation', 'test', 'baseline_test', 'compute', 'training'):
        if isinstance(report.get(key), dict): result[key] = scalars(report[key])
    if isinstance(report.get('dataset'), dict):
        result['dataset'] = {key: public_metadata(value) for key, value in report['dataset'].items()
                             if key in ('sample_count', 'split_counts', 'split_unit', 'supported_classes', 'min_train_positive_and_negative')}
    if isinstance(report.get('feature_contract'), dict):
        result['feature_contract'] = {key: public_metadata(value) for key, value in report['feature_contract'].items()
                                      if key in ('task', 'model_id', 'input_shape', 'reduction', 'raw_rgb_compatible',
                                                 'online_status', 'architecture', 'feature_dimension')}
    if isinstance(report.get('baselines'), dict):
        result['baselines'] = {name: {part: scalars(metrics) for part, metrics in records.items() if isinstance(metrics, dict)}
                               for name, records in report['baselines'].items() if isinstance(records, dict)}
    return public_metadata(result)


def aric_record(path):
    return {'scope': 'offline_research_only', 'online_compatible': False, 'report_sha256': sha256(path),
            'report': compact_metrics(read_json(path)),
            'limitation': 'ARIC特征输入与当前原始视频的在线合同未验证；该评测不是当前帧预测来源。'}


def context_evidence(context, index):
    labels = '、'.join(f"{row['name']} {row['probability']:.3f}" for row in context['labels']) or '没有类别超过验证阈值'
    return {'id': f'E{index:04d}', 'kind': 'behavior', 'start_sec': context['start_sec'],
            'end_sec': context['end_sec'], 'source_refs': [context['id']], 'method': 'trained_context',
            'source_split': context.get('source_split', 'unknown'),
            'text': f"{context['start_sec']:.2f}至{context['end_sec']:.2f}秒片段级教学情境预测：{labels}。模型{context['model_id']}；是训练分类器概率，不是逐秒事实或人工标注。"+context.get('text', '')}


def enrich_activity(activity, video_path, *, context_model=None, aric_report=None, device='cpu'):
    """Add trained model outputs to completed observations without rerunning detection."""
    if sha256(video_path) != activity.get('source_sha256'): raise ValueError('Enrichment video belongs to a different source')
    result = deepcopy(activity)
    if context_model:
        result['context_segments'], result['models']['context'] = load_context(video_path, context_model, device)
        result['evidence'] = [row for row in result['evidence'] if row.get('method') != 'trained_context']
        next_id = max((int(row['id'][1:]) for row in result['evidence'] if re.fullmatch(r'E\d+', row['id'])), default=0)+1
        result['evidence'].extend(context_evidence(row, next_id+i) for i, row in enumerate(result['context_segments']))
        result['limitations'] = [text for text in result['limitations'] if not text.startswith(('本轮未配置训练情境模型', '训练情境模型不可用', '教学情境为训练分类器'))]
        result['limitations'].append('教学情境为训练分类器对整个片段/窗口的粗粒度概率预测，不是逐秒真实行为或人工标注。')
    if aric_report:
        result['models']['aric'] = aric_record(aric_report)
        result['limitations'] = list(dict.fromkeys(result['limitations']+[result['models']['aric']['limitation']]))
    parts = urlsplit(result['frames'][0]['image_url']).path.split('/')
    validate_activity(result, {'lesson': {'id': parts[4], 'duration_sec': result['duration_sec'],
                                        'source_video': {'sha256': result['source_sha256']}}})
    return result


def frame_evidence(row, index):
    names = {'read': '阅读', 'write': '书写', 'raise_hand': '举手', 'unknown': '未知行为',
             'lookup':'模型抬头线索','turn_head':'转头线索','stand':'站立线索','discuss':'讨论线索'}
    observed = '、'.join(f"{names.get(label, label)}{amount}人次" for label, amount in row['behaviors'].items()) or '未检测到人物'
    score = '不可用' if row['participation_index'] is None else f"{row['participation_index']:.1f}/100（规则估计，不是概率）"
    coverage = f"{row['coverage']*100:.1f}%" if row['detected_people'] else '未知（无可用检测）'
    current = ''
    if row.get('participation'):
        p = row['participation']; names.update({'head_up':'头部抬起','head_lowered':'头部降低','hand_near_head':'手靠近头部','standing':'站姿线索'})
        distribution='、'.join(f"{names.get(label,label)}{amount}人" for label,amount in row['student_state_counts'].items())
        proportion=f"{p['active_fraction']*100:.1f}%" if p['active_fraction'] is not None else '未知'
        reference='不可用' if p['index'] is None else f"{p['index']:.1f}/100"
        weights='、'.join(f"{names.get(label,label)}{weight:g}" for label,weight in p['weights'].items())
        current=f"当前可见状态：{distribution or '无人可见'}；已检出参与动作{p['active_people']}/{p['total_people']}人，比例{proportion}。行为积极度参考{reference}（仅当前明确动作的规则均值；{weights}；支持{p['scored_people']}/{p['total_people']}人；不是心理积极度概率）。头部姿态仅为几何线索，不单独赋参与分；未识别不等于未参与。"
    if row.get('visual_reference'):
        ref=row['visual_reference']
        value='未知（支持不足）' if ref['index'] is None else f"{ref['index']:.1f}/100；平滑后{ref['smoothed_index']:.1f}/100"
        current+=f"七类模型的可见行为规则参考：{value}，可计分支持{ref['scored_people']}/{ref['total_people']}；采用原项目动作权重，不是心理积极度概率。转头、站立、未知和几何抬头不赋分；抬头类别不等于已确认听讲。"
    if row.get('detection_review'):
        current+='本轮先通过整图加重叠切片定位人物、融合姿态并去重，再逐框分类行为；分类器不生成新增人物框。检测框数不是全班人数或检测召回率；按本片段配置排除后墙反射候选。'
    return {'id': f'E{index:04d}', 'kind': 'behavior', 'start_sec': row['start_sec'], 'end_sec': row['end_sec'],
            'source_refs': [row['id']], 'image_url': row['image_url'], 'method': 'visible_behavior_observation',
            'text': f"{row['timestamp_sec']:.2f}秒抽样画面，镜头{row['scene_id']}：检测到{row['detected_people']}人（不推断师生身份）；{observed}。"
                    f"{current}具有可用时序规则分数{row['known_people']}人，覆盖率{coverage}；时序规则参与指数{score}。未知/不可见不能解释为低投入或知识未掌握。"}


def analyze_video(video_path, lesson_id, media_dir, *, interval_sec=1., device='cpu', context_model=None,
                  pose_config=None, aric_report=None, imgsz=640, scene_threshold=.22):
    """Analyze every scheduled sample of the full source, then publish one complete artifact."""
    import cv2
    import sys
    import time
    import uuid
    from datetime import datetime, timezone
    from .asr import inspect_video
    from .media import sample_times
    valid_id(lesson_id); number(interval_sec, .001); number(scene_threshold, .001, 1)
    if not isinstance(imgsz, int) or imgsz < 32: raise ValueError('Invalid inference image size')
    video_path = Path(video_path).resolve(); directory = Path(media_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = inspect_video(video_path); duration = source['duration_sec']
    if abs(source['video_start_sec']) > .01 or abs(source['video_duration_sec']-duration) > .05:
        raise ValueError('Activity source requires zero-offset, matching media/video duration')
    expected_bundle = {'lesson': {'id': lesson_id, 'duration_sec': duration, 'source_video': {'sha256': source['sha256']}}}
    if (directory/'bundle.json').is_file():
        active = read_json(directory/'bundle.json')['lesson']
        if active['id'] != lesson_id or active['source_video']['sha256'] != source['sha256']:
            raise ValueError('Activity and registered lesson belong to different sources')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    relative_frames = Path('activity/frames')/run_id; frames_dir = directory/relative_frames
    frames_dir.mkdir(parents=True)
    started = time.perf_counter(); started_at = datetime.now(timezone.utc).isoformat()
    config = Path(pose_config) if pose_config else Path(__file__).resolve().parents[2]/'vision/Class_Detection/configs/pipeline.yaml'
    pipeline = load_pipeline(config, device, imgsz)
    capture = cv2.VideoCapture(str(video_path)); fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0: raise ValueError('Video decode failed')
    pipeline.set_fps(fps)
    times = sample_times(duration, interval_sec)
    frames = []; snapshots = []; previous = None; scene = 1
    try:
        for index, timestamp in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
            ok, frame = capture.read()
            if not ok or frame is None: raise ValueError(f'Video sample could not be decoded at {timestamp}s')
            change = scene_difference(previous, frame) if previous is not None else 0.
            cut = previous is not None and change >= scene_threshold
            if cut:
                scene += 1; reset_tracking(pipeline)
            result = pipeline.process_frame(frame, frame_id=round(timestamp*fps), timestamp_sec=timestamp, enable_ocr=False)
            snapshot = result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
            filename = f'{index+1:06d}.jpg'; image = frames_dir/filename
            if not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]): raise ValueError('Activity frame could not be saved')
            end = times[index+1] if index+1 < len(times) else duration
            row = snapshot_frame(snapshot, index=index+1, timestamp=timestamp, end=end, scene_id=scene, scene_cut=cut,
                                 image_url=f'/api/review/media/{lesson_id}/{quote((relative_frames/filename).as_posix())}',
                                 width=frame.shape[1], height=frame.shape[0])
            row['scene_change_score'] = change
            frames.append(row); snapshot.update(scene_id=scene, scene_cut=cut, frame_image_path=row['image_url'])
            snapshots.append(snapshot); previous = frame
            print(f"Activity {index+1}/{len(times)} t={timestamp:.2f}s scene={scene} cut={cut} diff={change:.3f} people={row['detected_people']} tracks={len(row['participants'])} known={row['known_people']} labels={row['behaviors']}", flush=True)
    finally:
        capture.release()
    # Completed observation diagnostics remain recoverable if optional model metadata fails later.
    atomic_json(directory/'activity'/f'snapshots-v2-{run_id}.json', snapshots)
    limits = [
        '行为来自当前同源录像抽样；预训练阅读/书写/举手检测与姿态举手规则未在本课人工标注评测，不推断师生身份。',
        '行为参与指数是0–100的未校准规则估计，不是心理积极度概率；未知与不可见不记为低投入，不据此判断知识掌握或教学效果。',
        f'按{interval_sec:g}秒抽样并观察末尾画面；帧间状态与镜头边界只在抽样分辨率内近似，切镜头检测未经人工真值评测。每个镜头重置轨迹，轨迹不是跨镜头身份。',
        'known_people与coverage仅统计具有足够时序证据的规则分数；当前行为标签可见但时序证据不足时指数仍不可用。人次累计不是独立人物总数。']
    models = {'observation': public_metadata(pipeline.provenance),
              'scene_detection': {'method': 'downsampled_pixel_0.7_plus_gray_histogram_0.3', 'threshold': scene_threshold,
                                  'accuracy_evaluated': False},
              'runtime': {'device': device, 'imgsz': imgsz, 'python': sys.version.split()[0],
                          'source_pixel_width': source['width'], 'source_pixel_height': source['height']}}
    contexts = []
    if context_model:
        try: contexts, models['context'] = load_context(video_path, context_model, device)
        except Exception as exc:
            models['context'] = {'status': 'unavailable', 'reason': type(exc).__name__}
            limits.append('训练情境模型不可用；未使用参考标注替代预测。错误类型：'+type(exc).__name__)
        else: limits.append('教学情境为训练分类器对整个片段/窗口的粗粒度概率预测，不是逐秒真实行为或人工标注。')
    else: limits.append('本轮未配置训练情境模型，context_segments为空；未使用参考标注替代预测。')
    if aric_report:
        models['aric'] = aric_record(aric_report)
        limits.append(models['aric']['limitation'])
    evidence = [frame_evidence(row, index+1) for index, row in enumerate(frames)]
    for context in contexts:
        evidence.append(context_evidence(context, len(evidence)+1))
    activity = {'schema_version': '1.0', 'status': 'complete', 'source_sha256': source['sha256'], 'duration_sec': duration,
                'sampling_interval_sec': interval_sec, 'context_segments': contexts, 'frames': frames,
                'summary': {'sampled_frames': len(frames), 'scene_count': scene,
                            'person_observations': sum(r['detected_people'] for r in frames),
                            'known_observations': sum(r['known_people'] for r in frames)},
                'models': models, 'limitations': limits, 'evidence': evidence,
                'provenance': {'source_filename': source['filename'], 'started_at': started_at,
                               'completed_at': datetime.now(timezone.utc).isoformat(), 'wall_time_sec': round(time.perf_counter()-started, 3),
                               'expected_sample_count': len(times), 'time_base': 'source_media_seconds', 'timestamp_offset_sec': 0}}
    if sha256(video_path) != source['sha256']: raise ValueError('Source video changed during analysis')
    validate_activity(activity, expected_bundle, directory)
    if (directory/'bundle.json').is_file(): attach_activity(directory, activity)
    else:
        with LessonMutationLock(directory): atomic_json(directory/'activity.json', activity)
    return activity
