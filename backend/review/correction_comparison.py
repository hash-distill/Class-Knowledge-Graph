"""Optional comparison evidence bound to immutable source bytes and segment times."""
from copy import deepcopy
import math

from .common import read_json, sha256


def validate_comparison(data, bundle, asr_hash):
    if not isinstance(data, dict) or data.get('schema_version') != '1.0':
        raise ValueError('comparison_schema_invalid')
    if data.get('source_video_sha256') != bundle['lesson']['source_video']['sha256']:
        raise ValueError('comparison_video_changed')
    if data.get('source_asr_sha256') != asr_hash:
        raise ValueError('comparison_asr_changed')
    original = {r['id']: r for r in bundle['transcript']}
    runs = data.get('runs')
    if not isinstance(runs, list) or len(runs) != 2 or {r.get('id') for r in runs if isinstance(r, dict)} != {'qwen', 'cec4'}:
        raise ValueError('comparison_runs_invalid')
    for run in runs:
        for lane in ('segments', 'native_segments'):
            if lane == 'native_segments' and lane not in run:
                continue
            segments = run.get(lane)
            if not isinstance(segments, list):
                raise ValueError('comparison_segments_invalid')
            seen = set()
            for row in segments:
                if not isinstance(row, dict):
                    raise ValueError('comparison_segment_invalid')
                tid = row.get('transcript_id')
                raw = original.get(tid)
                if raw is None or tid in seen or row.get('original_text') != raw['text']:
                    raise ValueError('comparison_original_changed')
                for field in ('start_sec', 'end_sec'):
                    value = row.get(field)
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value - raw[field]) > 0.001:
                        raise ValueError('comparison_clock_changed')
                if not isinstance(row.get('text'), str) or not row['text'].strip():
                    raise ValueError('comparison_text_invalid')
                seen.add(tid)
            if seen != set(original):
                raise ValueError('comparison_incomplete_coverage')
    return data


def overlay_comparison(bundle, directory):
    path = directory/'correction.comparison.json'
    result = deepcopy(bundle)
    result.pop('correction_comparison', None)
    result.pop('correction_comparison_status', None)
    result['lesson']['provenance'].pop('correction_comparison', None)
    if not path.is_file():
        return result
    try:
        data = validate_comparison(read_json(path), bundle, sha256(directory/'asr.json'))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result['correction_comparison_status'] = {'status': 'unavailable', 'reason': str(exc)}
        return result
    result['correction_comparison'] = deepcopy(data)
    result['lesson']['provenance']['correction_comparison'] = {
        'artifact': path.name, 'sha256': sha256(path), 'scope': data.get('scope'),
        'reference_status': data.get('reference_status')}
    return result
