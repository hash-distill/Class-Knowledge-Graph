"""Synthetic fixtures for activity provenance, coverage and overlay regressions."""
import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.common import atomic_json
from review.retrieval import evidence_catalog


def fixture(tmp_path):
    image = tmp_path / 'activity/frames/run/000001.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'synthetic image fixture')
    bundle = {'schema_version': '3.0', 'data_source': 'lesson_review',
              'lesson': {'id': 'test', 'duration_sec': 2, 'source_video': {'sha256': 'a'*64},
                         'modalities': {'behavior': 'unavailable'}, 'provenance': {}},
              'transcript': [{'id': 'T0001', 'start_sec': 0, 'end_sec': 2, 'text': '测试语音'}],
              'slides': [], 'report': {'sections': [], 'limitations': [], 'statistics': {}}, 'snapshots': []}
    frames = [{'id': f'V{i+1:04d}', 'start_sec': i, 'end_sec': i+1, 'timestamp_sec': i,
               'scene_id': 1, 'scene_cut': False, 'image_url': '/api/review/media/test/activity/frames/run/000001.jpg',
               'status': 'low_evidence', 'detected_people': 1, 'known_people': 0, 'coverage': 0,
               'participation_index': None, 'behaviors': {'unknown': 1},
               'participants': [{'track_id': 'S1-T1', 'bbox': [0, 0, 10, 20], 'label': 'unknown',
                                 'confidence': 0, 'score': None, 'source': 'pose_rule'}]} for i in range(2)]
    activity = {'schema_version': '1.0', 'status': 'complete', 'source_sha256': 'a'*64,
                'duration_sec': 2, 'sampling_interval_sec': 1, 'context_segments': [], 'frames': frames,
                'summary': {'sampled_frames': 2, 'scene_count': 1, 'person_observations': 2, 'known_observations': 0},
                'models': {'observation': {'pipeline': 'fixture'}}, 'limitations': ['Synthetic test fixture.'],
                'evidence': [{'id': 'E0001', 'kind': 'behavior', 'start_sec': 0, 'end_sec': 1,
                              'source_refs': ['V0001'], 'text': '检测到1人，行为未知；规则参与指数不可用，覆盖率0%。'}]}
    return bundle, activity


def api():
    return importlib.import_module('review.activity')


def test_source_and_timing_validation_rejects_partial_or_unrelated_activity(tmp_path):
    bundle, activity = fixture(tmp_path)
    assert api().validate_activity(activity, bundle, tmp_path) == activity
    mutations = [lambda a: a.update(source_sha256='b'*64), lambda a: a.update(status='running'),
                 lambda a: a['frames'][0].update(end_sec=1.5),
                 lambda a: a['frames'][0].update(timestamp_sec=float('nan')),
                 lambda a: a['summary'].update(person_observations=3),
                 lambda a: a['frames'][0].update(image_url='https://outside/image.jpg'),
                 lambda a: a['frames'][0].update(image_url='/api/review/media/test/activity/../private.jpg'),
                 lambda a: a['frames'][0].update(coverage=.5),
                 lambda a: a['frames'][1].update(scene_id=2),
                 lambda a: a['evidence'][0].update(end_sec=2),
                 lambda a: a['evidence'][0].update(id='T0001')]
    for mutate in mutations:
        changed = copy.deepcopy(activity); mutate(changed)
        with pytest.raises(ValueError): api().validate_activity(changed, bundle, tmp_path)


def test_no_people_and_unknown_never_become_zero_participation(tmp_path):
    bundle, activity = fixture(tmp_path)
    activity['frames'][0].update(detected_people=0, participants=[], behaviors={}, status='no_people')
    activity['summary']['person_observations'] = 1
    assert api().validate_activity(activity, bundle, tmp_path)['frames'][0]['participation_index'] is None
    activity['frames'][0]['participation_index'] = 0
    with pytest.raises(ValueError): api().validate_activity(activity, bundle, tmp_path)


def test_overlay_is_copy_and_behavior_is_retrievable_with_original_source(tmp_path):
    bundle, activity = fixture(tmp_path); original = copy.deepcopy(bundle)
    result = api().overlay_activity(bundle, activity, tmp_path)
    assert bundle == original
    assert result['lesson']['modalities']['behavior'] == 'available'
    assert evidence_catalog(result)[-1]['kind'] == 'behavior'
    assert evidence_catalog(result)[-1]['source_refs'] == ['V0001']
    result['activity_analysis']['source_sha256'] = 'b'*64
    assert not [r for r in evidence_catalog(result) if r['kind'] == 'behavior']


def test_overlay_replaces_legacy_language_sections_without_mutating_the_source(tmp_path):
    bundle,activity=fixture(tmp_path)
    legacy_title='可见行为与课堂情境'
    legacy_limit='未提供与本课确认同步的学生行为录像，未计算投入度、知识掌握或教学效果。'
    bundle['report']['sections']=[{'title':legacy_title,'text':'Old generated behavior summary'},
        {'title':'Original source notes','text':'Keep this section'}]
    bundle['report']['limitations']=[legacy_limit,'Keep this independent limitation']
    before=copy.deepcopy(bundle)
    result=api().overlay_activity(bundle,activity,tmp_path)
    assert bundle==before
    assert [row['title'] for row in result['report']['sections']]==['Original source notes',api().REPORT_TITLE]
    assert legacy_limit not in result['report']['limitations']
    assert 'Keep this independent limitation' in result['report']['limitations']
    repeated=api().overlay_activity(result,activity,tmp_path)
    assert repeated['report']==result['report']


def test_snapshot_conversion_scales_scores_and_never_infers_roles():
    snapshot = {'student_states': [{'track_id': 7, 'bbox': [0, 0, 60, 100], 'cas': .8,
                 'action': {'label': 'read', 'confidence': .9, 'source': 'detection'},
                 'assessment': {'status': 'estimated'}}], 'classroom_metrics': {'untracked_people': 1}}
    row = api().snapshot_frame(snapshot, index=1, timestamp=0, end=1, scene_id=2, scene_cut=True,
                               image_url='/api/review/media/test/activity/frames/1.jpg', width=640, height=480)
    assert row['participants'][0]['track_id'] == 'S2-T7'
    assert row['participants'][0]['score'] == .8 and row['participation_index'] == 80
    assert row['coverage'] == .5 and row['detected_people'] == 2
    assert row['behaviors'] == {'read': 1, 'unknown': 1}
    assert 'role' not in row['participants'][0]
    snapshot['student_states'][0]['action']['label'] = 'unknown'
    row = api().snapshot_frame(snapshot, index=1, timestamp=0, end=1, scene_id=2, scene_cut=True,
                               image_url='/api/review/media/test/activity/frames/1.jpg', width=640, height=480)
    assert row['participation_index'] is None and row['known_people'] == 0


def test_courseware_removal_retains_new_activity_attached_after_import(tmp_path):
    from review.courseware_service import CoursewareService
    bundle, activity = fixture(tmp_path)
    bundle['lesson'].update(title='Synthetic fixture', status='complete')
    bundle['report'].update(title='Fixture', summary='Fixture')
    bundle.update(graph={'nodes': [], 'edges': []}, alignment=[])
    atomic_json(tmp_path/'bundle.base.json', bundle)
    current = api().overlay_activity(bundle, activity, tmp_path)
    current['courseware'] = {'document_id': 'Dtest'}
    atomic_json(tmp_path/'bundle.json', current)
    class Store:
        def entry(self, lesson_id): return {'directory': str(tmp_path)}
        def load(self, lesson_id): return copy.deepcopy(current)
    result = CoursewareService(Store()).remove('test')
    assert result['activity_analysis'] == activity
    assert result['lesson']['modalities']['behavior'] == 'available'


def test_builder_preserves_activity_json_on_rebuild(tmp_path, monkeypatch):
    import review.builder as builder
    bundle, activity = fixture(tmp_path)
    video = tmp_path/'source.mp4'; video.write_bytes(b'fixture source')
    from review.common import sha256
    activity['source_sha256'] = sha256(video)
    source = {'sha256': sha256(video)}
    atomic_json(tmp_path/'asr.json', {'status': 'complete', 'provenance': {'source_video': source},
                                    'segments': bundle['transcript']})
    image = tmp_path/'slide.jpg'; image.write_bytes(b'fixture')
    atomic_json(tmp_path/'slides.json', {'status': 'complete', 'provenance': {'source_video': source},
                                       'slides': [{'id': 'S0001', 'start_sec': 0, 'end_sec': 2, 'text': '测试', 'image_path': 'slide.jpg'}]})
    atomic_json(tmp_path/'activity.json', activity)
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    monkeypatch.setattr(builder, 'validate_media_timebase', lambda *a: 2)
    monkeypatch.setattr(builder, 'extract_concepts', lambda *a: {'nodes': [], 'edges': [], 'provenance': {}})
    class FixtureCorrector:
        def status(self): return {'source':'test_only'}
        def complete(self,*a,**k): return {'segments':[]},{'model':'fixture'}
    monkeypatch.setattr(builder, 'CorrectionProvider', FixtureCorrector)
    result = builder.build_lesson('test', video, tmp_path, 'Synthetic')
    assert result['activity_analysis'] == activity
    assert result['lesson']['modalities']['behavior'] == 'available'


def test_context_requires_trained_probabilities_and_original_media_time(tmp_path):
    bundle, activity = fixture(tmp_path)
    activity['context_segments'] = [{'id': 'X0001', 'start_sec': 0, 'end_sec': 2,
        'text': '片段级情境预测，非逐秒事实', 'model_id': 'test-trained', 'method': 'trained_context',
        'labels': [{'label': 'lecture', 'name': '讲授', 'probability': .7}]}]
    assert api().validate_activity(activity, bundle, tmp_path)
    activity['context_segments'][0]['labels'][0]['probability'] = 1.1
    with pytest.raises(ValueError): api().validate_activity(activity, bundle, tmp_path)


def test_model_metadata_handles_actual_integer_class_keys_without_private_paths():
    actual_shape = {'models': [{'names': {0: 'person'}, 'checkpoint_path': 'E:/private/model.pt'}],
                    'metrics': {'accuracy': .5}}
    result = api().public_metadata(actual_shape)
    assert result['models'][0]['names'] == {'0': 'person'}
    assert 'checkpoint_path' not in result['models'][0]


def test_context_abstention_is_valid_but_missing_behavior_evidence_is_not(tmp_path):
    bundle, activity = fixture(tmp_path)
    activity['context_segments'] = [{'id': 'X0001', 'start_sec': 0, 'end_sec': 2, 'labels': [],
        'text': 'No class exceeds the validation threshold', 'model_id': 'trained', 'method': 'trained_context'}]
    assert api().validate_activity(activity, bundle, tmp_path)
    activity['evidence'] = []
    with pytest.raises(ValueError): api().validate_activity(activity, bundle, tmp_path)


def test_scene_reset_clears_trackers_and_temporal_history():
    from types import SimpleNamespace
    tracker = SimpleNamespace(reset=lambda: reset_calls.append(True)); reset_calls = []
    pipeline = SimpleNamespace(pose=SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker])),
        evidence=SimpleNamespace(tracks={1: ['history']}), _last_timestamp=1, _track_epoch_offset=5, _max_emitted_track=9)
    api().reset_tracking(pipeline)
    assert reset_calls == [True] and pipeline.evidence.tracks == {}
    assert pipeline._last_timestamp is None and pipeline._track_epoch_offset == 0


def test_scene_difference_measures_pixels_and_cli_help_stays_lightweight():
    import subprocess
    np = pytest.importorskip('numpy'); pytest.importorskip('cv2')
    black = np.zeros((64, 64, 3), dtype=np.uint8); white = black+255
    assert api().scene_difference(black, black) == 0
    assert api().scene_difference(black, white) > .9
    script = Path(__file__).resolve().parents[1]/'scripts/analyze_activity.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--context-model' in result.stdout


def test_actual_video_sampling_with_test_only_pipeline_is_complete(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy'); cv2 = pytest.importorskip('cv2')
    video = tmp_path/'synthetic.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    for value in [0]*10+[255]*10: writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()
    class TestOnlyPipeline:
        provenance = {'pipeline': 'synthetic_test_only'}
        resets = 0
        def set_fps(self, fps): pass
        def process_frame(self, frame, **kwargs):
            return {'student_states': [], 'classroom_metrics': {'untracked_people': 0}}
    pipeline = TestOnlyPipeline()
    monkeypatch.setattr(api(), 'load_pipeline', lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(api(), 'reset_tracking', lambda p: setattr(p, 'resets', p.resets+1))
    result = api().analyze_video(video, 'test', tmp_path/'media', interval_sec=1)
    assert result['status'] == 'complete' and result['summary']['sampled_frames'] >= 2
    assert result['summary']['scene_count'] == 2 and pipeline.resets == 1
    assert result['frames'][-1]['end_sec'] == 2
    assert all(row['participation_index'] is None for row in result['frames'])
    assert (tmp_path/'media/activity.json').is_file()


def test_late_context_enrichment_keeps_observations_and_compacts_metrics(tmp_path, monkeypatch):
    from review.common import sha256
    bundle, activity = fixture(tmp_path)
    video = tmp_path/'source.mp4'; video.write_bytes(b'synthetic source')
    activity['source_sha256'] = sha256(video)
    context = {'id': 'X0001', 'start_sec': 0, 'end_sec': 2, 'text': '训练片段情境预测',
               'labels': [{'label': 'lecture', 'name': '讲授', 'probability': .7}],
               'model_id': 'test-model', 'method': 'trained_context'}
    monkeypatch.setattr(api(), 'load_context', lambda *args: ([context], {'scope': 'clip_level_prediction'}))
    aric = tmp_path/'metrics.json'
    atomic_json(aric, {'model_id': 'aric-test', 'dataset': {'sample_count': 50, 'split_counts': {'train': 30, 'test': 10},
                      'class_support': {'large': [1]*400}}, 'training': {'history': [1]*400, 'epochs_run': 20},
                      'test': {'macro_f1': .4, 'per_label': [1]*400}})
    result = api().enrich_activity(activity, video, context_model=tmp_path, aric_report=aric)
    assert result['frames'] == activity['frames'] and activity['context_segments'] == []
    assert result['context_segments'] == [context]
    assert result['evidence'][-1]['method'] == 'trained_context'
    assert result['models']['aric']['online_compatible'] is False
    assert len(str(result['models']['aric'])) < 1000
    video.write_bytes(b'changed source')
    with pytest.raises(ValueError): api().enrich_activity(activity, video, context_model=tmp_path)


@pytest.mark.parametrize('query',['全课学生行为复盘','Summarize student behavior across the whole lesson','LESSON OVERVIEW OF STUDENT PARTICIPATION'])
def test_whole_lesson_behavior_retrieval_covers_late_observations(tmp_path,query):
    from review.retrieval import select_evidence
    bundle, activity = fixture(tmp_path)
    bundle['lesson']['duration_sec'] = activity['duration_sec'] = 10
    frames = []
    for i in range(10):
        frame = copy.deepcopy(activity['frames'][0])
        frame.update(id=f'V{i+1:04d}', start_sec=i, end_sec=i+1, timestamp_sec=i)
        frames.append(frame)
    activity['frames'] = frames
    activity['summary'].update(sampled_frames=10, person_observations=10)
    activity['evidence'] = [api().frame_evidence(frame, i+1) for i, frame in enumerate(frames)]
    bundle['activity_analysis'] = activity
    bundle['slides'] = [{'id': 'S0001', 'start_sec': 0, 'end_sec': 10, 'text': '这是为整课查询提供内容概览的合成课件测试文本'}]
    rows = select_evidence(bundle, query, limit=6)
    assert any(r['kind'] == 'behavior' and r['start_sec'] == 9 for r in rows)


def test_behavior_answer_citations_include_actual_frame_source(tmp_path):
    from review.retrieval import answer_question
    bundle, activity = fixture(tmp_path)
    bundle['lesson']['title'] = 'Fixture'
    activity['evidence'] = [api().frame_evidence(activity['frames'][0], 1)]
    bundle['activity_analysis'] = activity
    class Provider:
        def complete(self, *args, **kwargs):
            return {'points': [{'text': '此刻行为未知。', 'evidence_ids': ['E0001']}]}, {'source': 'test_only', 'model': 'fixture'}
    result = answer_question(bundle, '此刻行为', 0, 'current', Provider())
    assert result['citations'][0]['source_refs'] == ['V0001']
    assert result['citations'][0]['image_url'] == activity['frames'][0]['image_url']
    assert result['citations'][0]['method'] == 'visible_behavior_observation'


def test_guided_voice_answer_exposes_source_verification_warning(tmp_path):
    from review.retrieval import answer_question
    bundle, _ = fixture(tmp_path)
    bundle['lesson']['title']='合成语音来源测试'
    bundle['transcript'][0]['quality']={'status':'machine_transcript_ocr_vocabulary_prompted_unreviewed'}
    class Provider:
        def complete(self,*args,**kwargs):
            return {'points':[{'text':'转写中的原句仍需核对。','evidence_ids':['T0001']}]},{'source':'test_only','model':'fixture'}
    result=answer_question(bundle,'语音内容',0,'lesson',Provider())
    assert any('Term transcription requires verification' in warning for warning in result['warnings'])
    bundle['transcript'][0].pop('quality')
    assert answer_question(bundle,'语音内容',0,'lesson',Provider())['warnings']==[]


def test_context_metadata_distinguishes_training_count_from_total_dataset(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'learning.ts_context', SimpleNamespace(predict_context=lambda *a, **k: []))
    atomic_json(tmp_path/'model.json', {'model_id': 'trained', 'training_count': 44, 'dataset_count': 70})
    atomic_json(tmp_path/'metrics.json', {'split_counts': {'train': 44, 'val': 10, 'test': 16}})
    _, metadata = api().load_context(tmp_path/'video.mp4', tmp_path, 'cpu')
    assert metadata['model']['training_count'] == 44
    assert metadata['model']['dataset_count'] == 70


def test_inference_publication_without_courseware_does_not_require_opencc(tmp_path):
    import subprocess
    bundle, activity = fixture(tmp_path)
    bundle['report'].update(title='Fixture', summary='Synthetic fixture')
    atomic_json(tmp_path/'bundle.json', bundle); atomic_json(tmp_path/'activity.json', activity)
    program = """import sys
sys.modules['opencc']=None
from review.activity import attach_activity
from review.common import read_json
from pathlib import Path
directory=Path(sys.argv[1])
result=attach_activity(directory,read_json(directory/'activity.json'))
assert result['lesson']['modalities']['behavior']=='available'
"""
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_context_citation_preserves_training_group_boundary():
    context = {'id': 'X0001', 'start_sec': 0, 'end_sec': 90, 'labels': [], 'model_id': 'trained',
               'source_split': 'train', 'text': '该课堂属于情境分类器训练组，仅用于功能演示，不作为留出验证。'}
    evidence = api().context_evidence(context, 182)
    assert evidence['source_split'] == 'train'
    assert context['text'] in evidence['text']


def test_no_detected_people_citation_has_unknown_coverage(tmp_path):
    _, activity = fixture(tmp_path)
    frame = activity['frames'][0]
    frame.update(detected_people=0, known_people=0, coverage=0, behaviors={}, participants=[], status='no_people')
    text = api().frame_evidence(frame, 1)['text']
    assert 'coverage unknown (no usable detections)' in text
    assert 'coverage 0.0%' not in text


@pytest.mark.parametrize('query',['请综合语音、课件和学生行为说明这段课堂。',
    'Compare speech, courseware and student behavior in this lesson.',
    'COMPARE AUDIO, SLIDES AND STUDENT PARTICIPATION.'])
def test_combined_question_keeps_voice_slide_and_observation_when_behavior_is_dense(tmp_path,query):
    from review.retrieval import select_evidence
    bundle, activity = fixture(tmp_path)
    bundle['lesson']['title'] = '三类证据合同夹具'
    bundle['slides'] = [{'id':'S0001','start_sec':0,'end_sec':2,'text':'认识氨基酸，观察其结构通式'}]
    bundle['transcript'][0]['text'] = '请写出通式，再独立思考。'
    activity['evidence'] = [{**activity['evidence'][0], 'id':f'E{i:04d}',
        'text':'Speech, courseware and student behavior. 语音课件学生行为，综合对照观察说明，规则参与指数未知。'} for i in range(1,31)]
    bundle['activity_analysis'] = activity
    rows = select_evidence(bundle, query, limit=3)
    assert {row['kind'] for row in rows} == {'transcript','slide','behavior'}
    assert len(rows) == 3
    bundle['transcript'] = [
        {'id':'T0001','start_sec':0,'end_sec':.3,'text':'请观察。','source':'asr'},
        {'id':'T0002','start_sec':.3,'end_sec':1.3,'text':'语音讲解通式。','source':'asr'},
        {'id':'T0003','start_sec':1.3,'end_sec':2,'text':'接着思考。','source':'asr'},
    ]
    rows = select_evidence(bundle, query, limit=3)
    assert len(rows) == 3, 'Neighbour context must respect the same retrieval budget'
