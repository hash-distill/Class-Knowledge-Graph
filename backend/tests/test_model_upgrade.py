from copy import deepcopy
import json

import pytest

from review.common import atomic_json, sha256
from review.model_upgrade import overlay_model_upgrade


@pytest.fixture
def measured(tmp_path):
    raw = {'id': 'T1', 'text': '认识氨基酸。', 'start_sec': 1, 'end_sec': 3}
    atomic_json(tmp_path/'asr.json', {'segments': [raw]})
    baseline_rows = [{'id': 'T1', 'text': '认识安基酸', 'start_sec': 0, 'end_sec': 4}]
    atomic_json(tmp_path/'asr.baseline.json', {'segments': baseline_rows})
    (tmp_path/'audio.wav').write_bytes(b'frozen-source-audio')
    digest = sha256(tmp_path/'asr.json')
    audio_hash = sha256(tmp_path/'audio.wav')
    row = {'transcript_id': 'T1', 'original_text': raw['text'], 'text': raw['text'],
           'start_sec': 1, 'end_sec': 3, 'uncertainties': [], 'edits': []}
    proof = {'segments': [row], 'edit_count': 0, 'status': 'complete', 'source_asr_sha256': digest,
             'provenance': {'model_label': 'Qwen3.5-4B', 'model_sha256': 'weights'}}
    atomic_json(tmp_path/'transcript.proofreading.json', proof)
    bundle = {'lesson': {'source_video': {'sha256': 'video'}, 'duration_sec': 10,
                         'provenance': {'asr': {'source_asr_sha256': digest, 'audio_sha256': audio_hash}}},
              'transcript': [{**raw, 'kind': 'transcript'}], 'transcript_proofreading': proof}
    data = {'schema_version': '1.0', 'source_video_sha256': 'video', 'source_audio_sha256': audio_hash,
            'reference_status': 'no_human_verbatim_reference', 'evaluated_at': '2026-10-28T01:00:00Z',
            'limitations': ['无人工逐字稿'], 'asr_runs': [
                {'id': 'baseline', 'model_label': '旧识别', 'source_asr_sha256': sha256(tmp_path/'asr.baseline.json'),
                 'segments': baseline_rows},
                {'id': 'qwen3_asr', 'model_label': 'Qwen3-ASR-1.7B', 'source_asr_sha256': digest,
                 'segments': [raw]}],
            'correction': {'model_label': 'Qwen3.5-4B', 'model_sha256': 'weights', 'source_asr_sha256': digest,
                           'status': 'complete', 'accepted_edit_count': 0, 'model_accepted_edit_count': 0,
                           'source_rule_accepted_count': 0,
                           'segments': [{k: v for k, v in row.items() if k != 'edits'}]}}
    atomic_json(tmp_path/'model.upgrade.json', data)
    return tmp_path, bundle, data


def test_independent_clocks_and_zero_correction_are_valid(measured):
    directory, bundle, data = measured
    before = deepcopy(bundle)
    result = overlay_model_upgrade(bundle, directory)
    assert result['model_upgrade_experiment'] == data
    assert result['lesson']['provenance']['model_upgrade_experiment']['sha256'] == sha256(directory/'model.upgrade.json')
    assert bundle == before


@pytest.mark.parametrize('change', [
    lambda d: d.update(source_video_sha256='changed'),
    lambda d: d.update(source_audio_sha256='changed'),
    lambda d: d.update(reference_status='human_verified'),
    lambda d: d['asr_runs'][1].update(source_asr_sha256='changed'),
    lambda d: d['asr_runs'][1]['segments'][0].update(text='补写'),
    lambda d: d['asr_runs'][0]['segments'][0].update(text='伪造旧识别'),
    lambda d: d['asr_runs'][0]['segments'][0].update(end_sec=float('inf')),
    lambda d: d['correction'].update(accepted_edit_count=1),
    lambda d: d['correction'].update(model_accepted_edit_count=True),
    lambda d: d['correction'].update(model_sha256='changed'),
    lambda d: d['correction']['segments'][0].update(uncertainties=['待复核']),
    lambda d: d['correction'].update(segments=[]),
])
def test_bad_optional_evidence_clears_stale_payload(measured, change):
    directory, bundle, data = measured
    previous = overlay_model_upgrade(bundle, directory)
    change(data)
    (directory/'model.upgrade.json').write_text(json.dumps(data), encoding='utf-8')
    result = overlay_model_upgrade(previous, directory)
    assert 'model_upgrade_experiment' not in result
    assert 'model_upgrade_experiment' not in result['lesson']['provenance']
    assert result['model_upgrade_status']['status'] == 'unavailable'
    assert result['model_upgrade_status']['status'] == 'unavailable'


@pytest.mark.parametrize('file', ['asr.json', 'asr.baseline.json', 'transcript.proofreading.json', 'audio.wav'])
def test_saved_artifact_mutation_is_rejected(measured, file):
    directory, bundle, _ = measured
    if file == 'audio.wav':
        (directory/file).write_bytes(b'changed')
    else:
        atomic_json(directory/file, {})
    result = overlay_model_upgrade(bundle, directory)
    assert result['model_upgrade_status']['status'] == 'unavailable'


def test_removed_optional_artifact_removes_previous_payload(measured):
    directory, bundle, _ = measured
    previous = overlay_model_upgrade(bundle, directory)
    (directory/'model.upgrade.json').unlink()
    result = overlay_model_upgrade(previous, directory)
    assert 'model_upgrade_experiment' not in result
    assert 'model_upgrade_experiment' not in result['lesson']['provenance']


def test_api_cache_revalidates_optional_evidence_when_source_changes(measured, tmp_path):
    from review.server import LessonStore
    directory, bundle, _ = measured
    bundle.update(schema_version='3.0', data_source='lesson_review')
    bundle['lesson'].update(id='example', status='complete')
    atomic_json(directory/'bundle.json', overlay_model_upgrade(bundle, directory))
    registry = tmp_path/'registry.json'
    atomic_json(registry, {'example': {'directory': str(directory)}})
    store = LessonStore(registry)
    assert 'model_upgrade_experiment' in store.load('example')
    (directory/'audio.wav').write_bytes(b'changed-audio')
    assert 'model_upgrade_experiment' not in store.load('example')
    assert store.load('example')['model_upgrade_status']['status'] == 'unavailable'
