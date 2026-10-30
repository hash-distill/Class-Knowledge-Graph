"""Reproducible pairing, independent scoring and failure transparency."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_qwen_upgrade import (ASR_KEYS, CORRECTOR_KEYS, NATIVE_SYSTEM,
    PRIMARY_ID, build_plan, build_report, correction_path, file_sha256,
    load_asr, run_corrections, save, text_sha256, write_report)
from review.provider import ReviewError


def fixture_audio(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(100)
        stream.writeframes(b'\0\0'*200)


def asr_file(path, audio_sha, rows, legacy=False):
    provenance = {'audio_sha256': audio_sha} if legacy else {
        'source_audio': {'sha256': audio_sha}, 'time_base': 'source_audio_seconds'}
    save(path, {'status': 'complete', 'segments': rows, 'provenance': provenance})


def rows(texts):
    return [{'id': f'T{i}', 'start_sec': 2*i/len(texts), 'end_sec': 2*(i+1)/len(texts), 'text': text}
            for i, text in enumerate(texts)]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    primary, external, upgrade = (tmp_path/k for k in ('primary', 'external', 'upgrade'))
    fixture_audio(primary/'audio.wav')
    audio_sha = file_sha256(primary/'audio.wav')
    save(primary/'audio.provenance.json', {'sha256': audio_sha})
    asr_file(primary/'asr.json', audio_sha, rows(['课堂', '口语啊']), legacy=True)
    asr_file(primary/'asr.unguided.json', audio_sha, rows(['课躺口语啊']), legacy=True)
    asr_file(upgrade/'asr'/PRIMARY_ID/'asr.json', audio_sha, rows(['课堂口语啊']))
    sample_rows, inputs, gold = [], [], []
    for identifier in ('external-a', 'external-b'):
        wav = external/'samples'/identifier/'audio.wav'; fixture_audio(wav)
        sha = file_sha256(wav)
        annotation = wav.parent/'official-annotation.json'
        save(annotation, {'paragraphs': {'content': '甲乙丙'}})
        sample_rows.append({'id': identifier, 'files': {'wav': {'path': str(wav), 'sha256': sha},
            'annotation': {'path': str(annotation), 'sha256': file_sha256(annotation)}}})
        inputs.append({'id': identifier, 'wav_path': str(wav), 'ground_truth_in_model_inputs': False, 'reference_fields_excluded': True})
        gold.append({'id': identifier, 'source': 'official_manual_transcription', 'reference_text': '甲乙丙',
                     'source_annotation_sha256': file_sha256(annotation)})
        asr_file(external/'evaluation'/identifier/'asr.json', sha, rows(['甲错丙']))
        # Different T IDs/segmentation must still score the complete same WAV.
        asr_file(upgrade/'asr'/identifier/'asr.json', sha, rows(['甲', '乙', '丙']))
        save(external/'evaluation'/identifier/'reference.json', {
            'reference_kind': 'official_verbatim', 'human_verified': True, 'independent_of_model_outputs': True,
            'used_as_model_input': False, 'time_base': 'source_audio_seconds', 'source_audio_sha256': sha,
            'provenance': {'source': 'https://example.test/pinned-manual-source'},
            'segments': [{'transcript_id': 'OFFICIAL', 'start_sec': 0., 'end_sec': 2., 'text': '甲乙丙'}]})
    save(external/'manifest.json', {'samples': sample_rows})
    save(external/'model-inputs.json', {'samples': inputs})
    save(external/'gold-references.json', {'use': 'scorer_only_do_not_send_to_models', 'samples': gold})
    model = tmp_path/'test-model.gguf'; model.write_bytes(b'not a model: deterministic fake provider fixture')
    profile = tmp_path/'profile.json'
    save(profile, {'base_url': 'http://127.0.0.1:1/v1', 'model_alias': 'test-only',
        'model': {'path': str(model), 'sha256': file_sha256(model)}, 'server': {'context_size': 4096}})
    monkeypatch.setattr('scripts.compare_qwen_upgrade.expected_profile', lambda key: json.loads(profile.read_text(encoding='utf-8')))
    plan = build_plan(primary, external, upgrade)
    return plan, primary, external, upgrade, profile


class FakeProvider:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def complete(self, system, prompt, *, max_tokens):
        self.calls.append((system, prompt, max_tokens))
        if self.fail:
            raise ReviewError('test_failure', 'fixture failure', diagnostic={'raw_response': 'partial'})
        text = prompt.replace('错', '乙')
        return text, {'raw_response': text, 'finish_reason': 'stop', 'request_parameters': {'max_tokens': max_tokens}}


def populate(plan, output_dir, profile):
    for asr in ASR_KEYS:
        for corrector in CORRECTOR_KEYS:
            run_corrections(plan, asr, corrector, profile, output_dir, provider=FakeProvider())


def test_prepare_and_run_do_not_open_or_embed_scorer_files(experiment):
    plan, primary, external, upgrade, profile = experiment
    (external/'gold-references.json').write_text('not JSON', encoding='utf-8')
    for path in (external/'evaluation').glob('*/reference.json'):
        path.write_text('not JSON', encoding='utf-8')
    prepared = build_plan(primary, external, upgrade)
    assert prepared['generation_input_policy']['scorer_files_read'] is False
    assert 'reference_text' not in json.dumps(prepared['samples'])
    provider = FakeProvider()
    records = run_corrections(prepared, 'old', 'qwen35', profile, upgrade/'evaluation', provider=provider)
    assert len(records) == len(prepared['samples'])
    assert all(call[0] == NATIVE_SYSTEM and call[2] == 2000 for call in provider.calls)
    assert [call[1] for call in provider.calls] == [s['asr_runs']['old']['text'] for s in plan['samples']]
    assert all(r['provenance']['gold_or_reference_read'] is False for r in records)


def test_whole_clip_scoring_joins_asr_without_id_or_segmentation_alignment(experiment):
    plan, _, external, upgrade, profile = experiment
    output = upgrade/'evaluation'; populate(plan, output, profile)
    report = build_report(plan, output, external)
    assert report['status'] == 'complete'
    assert report['samples'][0]['accuracy_evaluated'] is False
    assert all(r['score']['reference_metrics'] is None for r in report['samples'][0]['correction_runs'])
    first = report['samples'][1]
    assert first['asr_runs']['old']['segment_count'] == 1
    assert first['asr_runs']['qwen3']['segment_count'] == 3
    old = next(r for r in first['correction_runs'] if r['asr_key'] == 'old')
    assert old['score']['reference_metrics']['strict']['cer'] == 0
    assert old['score']['normalized_reference_character_transitions']['repaired_reference_characters'] == 1
    assert report['external_summary']['paired_complete_sample_ids'] == ['external-a', 'external-b']
    assert report['external_summary']['cells']['old:qwen35']['reference_metrics']['strict']['reference_characters'] == 6
    write_report(report, output)
    assert all((output/name).is_file() for name in ('comparison.json', 'comparison.md', 'segments.csv'))


def test_failed_and_missing_outputs_are_unknown_and_excluded_from_all_cell_pair(experiment):
    plan, _, external, upgrade, profile = experiment
    output = upgrade/'evaluation'; populate(plan, output, profile)
    missing = correction_path(output, 'old', 'old_qwen', 'external-b'); missing.unlink()
    provider = FakeProvider(fail=True)
    record = run_corrections(plan, 'old', 'old_qwen', profile, output, provider=provider)[-1]
    assert record['status'] == 'failed' and record['output_text'] is None
    assert record['raw_response_sha256'] == text_sha256('partial')
    report = build_report(plan, output, external)
    failed = report['samples'][-1]['correction_runs'][0]
    assert failed['status'] == 'failed' and failed['score'] is None and failed['output_text'] is None
    assert report['external_summary']['paired_complete_sample_ids'] == ['external-a']
    assert report['external_summary']['excluded_sample_ids'] == ['external-b']
    assert all(c['reference_metrics']['strict']['reference_characters'] == 3 for c in report['external_summary']['cells'].values())


def test_missing_new_asr_is_unavailable_never_empty_hypothesis(experiment):
    plan, _, external, upgrade, _ = experiment
    missing_path = Path(plan['samples'][1]['asr_runs']['qwen3']['path']); missing_path.unlink()
    asr = load_asr(missing_path, plan['samples'][1]['source_audio'])
    assert asr['status'] == 'missing' and asr['text'] is None
    plan['samples'][1]['asr_runs']['qwen3'] = asr
    report = build_report(plan, upgrade/'evaluation', external)
    assert report['samples'][1]['asr_scores']['qwen3'] is None
    assert report['external_summary']['paired_complete_sample_ids'] == []
    assert all(c['reference_metrics']['strict']['cer'] is None for c in report['external_summary']['cells'].values())


@pytest.mark.parametrize('corruption', ['audio', 'asr', 'reference', 'annotation', 'conditions'])
def test_detects_audio_asr_reference_annotation_or_condition_drift(experiment, corruption):
    plan, _, external, upgrade, profile = experiment
    output = upgrade/'evaluation'; populate(plan, output, profile)
    sample = plan['samples'][1]
    if corruption == 'audio':
        Path(sample['source_audio']['path']).write_bytes(b'changed')
    elif corruption == 'asr':
        path = Path(sample['asr_runs']['qwen3']['path']); path.write_text('{}', encoding='utf-8')
    elif corruption == 'reference':
        path = external/'evaluation/external-a/reference.json'; value = json.loads(path.read_text()); value['segments'][0]['text'] = '伪造'; save(path, value)
    elif corruption == 'annotation':
        save(external/'samples/external-a/official-annotation.json', {'paragraphs': {'content': '伪造'}})
    else:
        path = correction_path(output, 'old', 'old_qwen', 'external-a'); value = json.loads(path.read_text()); value['conditions']['max_tokens'] = 5; save(path, value)
    with pytest.raises(ValueError):
        build_report(plan, output, external)


def test_resume_reuses_only_matching_contract_and_fixed_policy(experiment):
    plan, _, _, upgrade, profile = experiment
    output = upgrade/'evaluation'; provider = FakeProvider()
    run_corrections(plan, 'old', 'old_qwen', profile, output, provider=provider)
    assert len(provider.calls) == 3
    run_corrections(plan, 'old', 'old_qwen', profile, output, provider=provider)
    assert len(provider.calls) == 3
    changed = deepcopy(plan); changed['generation_input_policy']['max_tokens'] = 1
    with pytest.raises(ValueError, match='fixed conservative'):
        run_corrections(changed, 'old', 'old_qwen', profile, output, provider=provider)


def test_reference_damage_and_insertions_have_separate_denominators(experiment):
    plan, _, external, upgrade, profile = experiment
    class DamageProvider(FakeProvider):
        def complete(self, system, prompt, *, max_tokens):
            return '甲乙坏新增', {'finish_reason': 'stop'}
    output = upgrade/'evaluation'
    run_corrections(plan, 'old', 'qwen35', profile, output, provider=DamageProvider())
    report = build_report(plan, output, external)
    run = next(r for r in report['samples'][1]['correction_runs'] if r['corrector_key'] == 'qwen35' and r['asr_key'] == 'old')
    transitions = run['score']['normalized_reference_character_transitions']
    assert transitions['reference_characters'] == 3
    assert transitions['initially_correct_reference_characters'] == 2
    assert transitions['initially_incorrect_reference_characters'] == 1
    assert transitions['repaired_reference_characters'] == 1
    assert transitions['damaged_previously_correct_reference_characters'] == 1
    assert transitions['insertion_errors_after'] == 2


def test_published_wrapper_uses_actual_native_text_and_binds_guard_hash(experiment):
    _, primary, external, upgrade, _ = experiment
    native_path = upgrade/'asr'/PRIMARY_ID/'asr.json'
    wrapper = json.loads(native_path.read_text(encoding='utf-8'))
    wrapper['provenance']['native_asr_artifact'] = {'sha256': file_sha256(native_path)}
    path = upgrade/'published/asr.json'
    save(path, wrapper)
    plan = build_plan(primary, external, upgrade, path)
    assert plan['samples'][0]['asr_runs']['qwen3']['sha256'] == file_sha256(path)
    wrapper['segments'][0]['text'] = '伪造新识别'
    save(path, wrapper)
    with pytest.raises(ValueError, match='differs from actual native'):
        build_plan(primary, external, upgrade, path)


def test_wrong_corrector_profile_is_rejected_before_generation(experiment):
    plan, _, _, upgrade, profile = experiment
    different = json.loads(profile.read_text(encoding='utf-8'))
    different['model_alias'] = 'wrong-corrector'
    path = upgrade/'wrong-profile.json'
    save(path, different)
    provider = FakeProvider()
    with pytest.raises(ValueError, match='alias does not match'):
        run_corrections(plan, 'old', 'qwen35', path, upgrade/'evaluation', provider=provider)
    assert not provider.calls


def test_report_rejects_tampered_model_contract(experiment):
    plan, _, external, upgrade, profile = experiment
    output = upgrade/'evaluation'
    run_corrections(plan, 'old', 'old_qwen', profile, output, provider=FakeProvider())
    path = correction_path(output, 'old', 'old_qwen', PRIMARY_ID)
    record = json.loads(path.read_text(encoding='utf-8'))
    record['benchmark_contract']['model_sha256'] = 'wrong-model'
    save(path, record)
    with pytest.raises(ValueError, match='model/profile contract differs'):
        build_report(plan, output, external)
