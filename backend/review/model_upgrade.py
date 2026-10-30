"""Optional measured model-upgrade evidence, bound to actual lesson artifacts."""
from copy import deepcopy
from datetime import datetime
import math

from .common import read_json, sha256


def _require(condition, reason):
    if not condition:
        raise ValueError('model_upgrade_' + reason)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _rows(rows, duration):
    _require(isinstance(rows, list) and bool(rows), 'segments_missing')
    seen = set()
    for row in rows:
        _require(isinstance(row, dict), 'segment_invalid')
        ident = row.get('id')
        _require(_text(ident) and ident not in seen and _text(row.get('text')), 'segment_invalid')
        start, end = row.get('start_sec'), row.get('end_sec')
        _require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                     for v in (start, end)), 'clock_invalid')
        _require(0 <= start <= end <= duration, 'clock_invalid')
        seen.add(ident)
    return {row['id']: row for row in rows}


def validate_model_upgrade(data, bundle, asr_hash, audio_hash):
    _require(isinstance(data, dict) and data.get('schema_version') == '1.0', 'schema_invalid')
    _require(data.get('reference_status') == 'no_human_verbatim_reference', 'reference_invalid')
    lesson = bundle['lesson']
    _require(data.get('source_video_sha256') == lesson['source_video']['sha256'], 'video_changed')
    _require(data.get('source_audio_sha256') == audio_hash, 'audio_changed')
    provenance = lesson['provenance']['asr']
    _require(provenance.get('source_asr_sha256') == asr_hash, 'bundle_asr_changed')
    _require((provenance.get('audio_sha256') or provenance.get('source_audio', {}).get('sha256')) == audio_hash, 'bundle_audio_changed')
    datetime.fromisoformat(data['evaluated_at'].replace('Z', '+00:00'))
    _require(isinstance(data.get('limitations'), list) and all(_text(v) for v in data['limitations']), 'limits_invalid')
    native = _rows(bundle['transcript'], lesson['duration_sec'])
    runs = data.get('asr_runs')
    _require(isinstance(runs, list) and len(runs) == 2, 'runs_invalid')
    seen = set()
    for run in runs:
        _require(isinstance(run, dict), 'run_invalid')
        ident = run.get('id')
        _require(ident in ('baseline', 'qwen3_asr') and ident not in seen, 'runs_invalid')
        _require(_text(run.get('model_label')) and _text(run.get('source_asr_sha256')), 'run_source_invalid')
        rows = _rows(run.get('segments'), lesson['duration_sec'])
        if ident == 'qwen3_asr':
            _require(run['source_asr_sha256'] == asr_hash and rows.keys() == native.keys(), 'native_source_changed')
            for key, row in rows.items():
                _require(all(row.get(f) == native[key].get(f) for f in ('text', 'start_sec', 'end_sec')), 'native_text_changed')
        seen.add(ident)
    proof = bundle['transcript_proofreading']
    correction = data.get('correction')
    _require(isinstance(correction, dict), 'correction_missing')
    _require(proof.get('source_asr_sha256') == asr_hash and correction.get('source_asr_sha256') == asr_hash, 'proof_source_changed')
    _require(_text(correction.get('model_label')) and _text(correction.get('model_sha256')), 'model_missing')
    _require(correction.get('status') == proof.get('status'), 'proof_status_changed')
    for field in ('model_label', 'model_sha256'):
        _require(correction[field] == proof['provenance'].get(field), 'proof_model_changed')
    edits = [edit for row in proof['segments'] for edit in row['edits']]
    model_count = sum(e.get('origin') == 'model' for e in edits)
    for field, expected in [('accepted_edit_count', len(edits)), ('model_accepted_edit_count', model_count),
                            ('source_rule_accepted_count', len(edits) - model_count)]:
        value = correction.get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value == expected, 'edit_count_changed')
    _require(proof['edit_count'] == len(edits), 'proof_count_changed')
    expected = {r['transcript_id']: r for r in proof['segments']}
    rows = correction.get('segments')
    _require(isinstance(rows, list) and len(rows) == len(expected), 'proof_coverage_changed')
    seen = set()
    for row in rows:
        _require(isinstance(row, dict), 'proof_row_invalid')
        key = row.get('transcript_id')
        _require(key in expected and key not in seen, 'proof_coverage_changed')
        _require(all(row.get(f) == expected[key].get(f) for f in
                     ('original_text', 'text', 'start_sec', 'end_sec', 'uncertainties')), 'proof_text_changed')
        seen.add(key)
    return data


def overlay_model_upgrade(bundle, directory):
    result = deepcopy(bundle)
    result.pop('model_upgrade_experiment', None)
    result.pop('model_upgrade_status', None)
    result['lesson']['provenance'].pop('model_upgrade_experiment', None)
    path = directory/'model.upgrade.json'
    if not path.is_file():
        if 'model_upgrade_experiment' in bundle:
            result['model_upgrade_status'] = {'status': 'unavailable', 'reason': 'model_upgrade_artifact_missing'}
        return result
    try:
        # Stale bundle proof must not validate a different saved correction.
        _require(read_json(directory/'transcript.proofreading.json') == bundle['transcript_proofreading'], 'saved_proof_changed')
        raw = read_json(directory/'asr.json')
        raw_rows = {r['id']: r for r in raw['segments']}
        published_rows = {r['id']: r for r in bundle['transcript']}
        _require(raw_rows.keys() == published_rows.keys() and all(
            all(row.get(f) == published_rows[key].get(f) for f in ('text', 'start_sec', 'end_sec'))
            for key, row in raw_rows.items()), 'saved_transcript_changed')
        data = validate_model_upgrade(read_json(path), bundle, sha256(directory/'asr.json'), sha256(directory/'audio.wav'))
        baseline_path = directory/'asr.baseline.json'
        baseline = read_json(baseline_path)
        run = next(r for r in data['asr_runs'] if r['id'] == 'baseline')
        _require(run['source_asr_sha256'] == sha256(baseline_path), 'baseline_source_changed')
        expected_baseline = {r['id']: r for r in baseline['segments']}
        published_baseline = {r['id']: r for r in run['segments']}
        _require(expected_baseline.keys() == published_baseline.keys() and all(
            all(row.get(f) == published_baseline[key].get(f) for f in ('text', 'start_sec', 'end_sec'))
            for key, row in expected_baseline.items()), 'baseline_text_changed')
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        result['model_upgrade_status'] = {'status': 'unavailable', 'reason': str(exc)}
        return result
    result['model_upgrade_experiment'] = deepcopy(data)
    result['lesson']['provenance']['model_upgrade_experiment'] = {
        'artifact': path.name, 'sha256': sha256(path), 'reference_status': data['reference_status']}
    return result
