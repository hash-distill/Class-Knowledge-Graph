"""Benchmark bookkeeping checks; all model/GPU responses here are synthetic."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.provider import ReviewError
from scripts.run_correction_benchmark import combine_runtime_attempts, native_run, parse_gpu_csv, source_identity


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')


def test_gpu_parser_retains_whole_device_units_and_card_identity():
    row = parse_gpu_csv('0, Test GPU, GPU-test, 8192, 4000, 95, 999.0, 42.50\n')
    assert row['memory_used_mib'] == 4000 and row['memory_total_mib'] == 8192
    assert row['uuid'] == 'GPU-test' and row['power_w'] == 42.5
    with pytest.raises(ValueError):
        parse_gpu_csv('0, Test GPU\n')


def test_native_error_is_audited_and_original_is_preserved(tmp_path):
    media = tmp_path/'media'; media.mkdir()
    save(media/'asr.json', {'status': 'complete', 'provenance': {'source_audio': {'sha256': 'a'*64}},
        'segments': [{'id': 'T0001', 'start_sec': 0., 'end_sec': 2., 'text': '测试草稿'}]})
    profile = tmp_path/'profile.json'
    config = {'model': {'sha256': 'b'*64}, 'request_defaults': {'temperature': .1}, 'server': {'context_size': 24576}}
    save(profile, config)
    class Provider:
        model = 'test_only'
        def __init__(self):
            self.config = config
            self.calls = 0
        def native_correct(self, text, **kwargs):
            self.calls += 1
            raise ReviewError('model_incomplete', 'synthetic truncation', diagnostic={
                'raw_response': '<think>unfinished synthetic response', 'finish_reason': 'length', 'usage': {'completion_tokens': 1000}})
    class Sampler:
        def summarize(self, *args):
            return {'scope': 'test_only_synthetic_no_gpu_measurement'}
    provider = Provider(); path = tmp_path/'native.json'
    result = native_run(media, path, provider, 'test', profile, 1000, Sampler(), 'baseline', 'loaded', 0, 'native')
    assert result['result']['status'] == 'partial'
    assert result['native_failure_count'] == 1
    assert result['native_outputs'][0]['corrected_text'] == '测试草稿'
    assert result['native_outputs'][0]['metadata']['finish_reason'] == 'length'
    assert result['result']['source_audio_sha256'] == 'a'*64
    assert 'source_video_sha256' not in result['result']
    native_run(media, path, provider, 'test', profile, 1000, Sampler(), 'baseline', 'loaded', 0, 'native')
    assert provider.calls == 1


def test_native_source_cannot_be_guessed_without_real_identity(tmp_path):
    save(tmp_path/'asr.json', {'status': 'complete', 'segments': [], 'provenance': {}})
    with pytest.raises(ValueError, match='source video/audio'):
        source_identity(tmp_path)


def test_resumed_native_peak_and_time_keep_all_measured_attempts():
    combined = combine_runtime_attempts([
        {'peak_device_used_mib': 6000, 'generation_wall_time_sec': 30, 'sample_file': 'attempt1.jsonl', 'sample_count': 100},
        {'peak_device_used_mib': 5500, 'generation_wall_time_sec': 10, 'sample_file': 'attempt2.jsonl', 'sample_count': 40}])
    assert combined['peak_device_used_mib'] == 6000
    assert combined['generation_wall_time_sec'] == 40
    assert combined['sample_files'] == ['attempt1.jsonl', 'attempt2.jsonl']
    assert combined['sample_count'] == 140
