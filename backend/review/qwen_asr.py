"""Evidence-preserving conversion of native Qwen ASR and forced-aligner output.

This module has no model dependencies. Segment boundaries always come from real
aligner predictions; unavailable timestamps produce one explicitly labelled clip.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def inspect_wav(path, expected_sha256=None):
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('audio_sha256_mismatch: ' + path.name)
    with wave.open(str(path), 'rb') as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        result = {'filename': path.name, 'sha256': digest,
                  'size_bytes': path.stat().st_size, 'sample_rate': rate,
                  'frame_count': frames, 'channels': audio.getnchannels(),
                  'sample_width_bytes': audio.getsampwidth(),
                  'duration_sec': frames / rate, 'audio_start_sec': 0.0}
    if result['duration_sec'] <= 0:
        raise ValueError('empty_audio')
    return result


def _span(item):
    text = str(item['text'])
    start, end = float(item['start_time']), float(item['end_time'])
    if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end < start:
        raise ValueError('invalid_native_timestamp')
    return text, start, end


def native_segments(text, items, duration):
    """Group native tokens at existing punctuation/gaps, preserving exact text.

    No estimated character times, silence trimming, or equal-width splitting is
    used. Punctuation belongs to the preceding native span and gets no new time.
    """
    if not items:
        return ([{'id': 'T0001', 'start_sec': 0.0, 'end_sec': duration,
                  'text': text, 'source': 'qwen3_asr',
                  'timestamp_method': 'whole_clip_no_predicted_timestamps',
                  'quality': {'status': 'machine_transcript_unreviewed',
                              'timing_status': 'whole_clip_only'}}] if text else [])
    spans, cursor = [], 0
    for index, item in enumerate(items):
        token, start, end = _span(item)
        if end > duration:
            raise ValueError('native_timestamp_outside_audio')
        position = text.find(token, cursor)
        if not token or position < 0:
            raise ValueError('native_token_not_found_in_transcript')
        if spans and start < spans[-1]['start']:
            raise ValueError('native_timestamps_not_monotonic')
        spans.append({'start': start, 'end': end, 'char_start': position,
                      'char_end': position + len(token), 'item_index': index})
        cursor = position + len(token)
    groups, first = [], 0
    for index, span in enumerate(spans):
        last = index == len(spans) - 1
        next_char = len(text) if last else spans[index + 1]['char_start']
        punctuation = text[span['char_end']:next_char]
        gap = 0 if last else spans[index + 1]['start'] - span['end']
        width = span['end'] - spans[first]['start']
        if last or any(x in punctuation for x in '。！？!?；;') or gap >= .8 or width >= 9:
            # Zero-duration native units are merged with their neighbours without
            # changing their predicted values.
            if span['end'] > spans[first]['start']:
                groups.append((first, index, next_char))
                first = index + 1
    if first < len(spans):
        if not groups:
            raise ValueError('all_native_timestamps_have_zero_duration')
        a, _, _ = groups[-1]
        groups[-1] = (a, len(spans) - 1, len(text))
    result, char_cursor = [], 0
    for a, b, char_end in groups:
        result.append({'id': f'T{len(result)+1:04d}',
                       'start_sec': spans[a]['start'], 'end_sec': spans[b]['end'],
                       'text': text[char_cursor:char_end], 'source': 'qwen3_asr',
                       'timestamp_method': 'qwen3_forced_aligner_predicted',
                       'native_alignment_item_range': [a, b],
                       'quality': {'status': 'machine_transcript_unreviewed',
                                   'timing_status': 'predicted_unreviewed'}})
        char_cursor = char_end
    if ''.join(row['text'] for row in result) != text:
        raise ValueError('segmentation_changed_native_transcript')
    return result


def build_asr(native, audio, config, source_video=None, metrics=None):
    text = native['text']
    alignment = native.get('time_stamps') or {}
    items = alignment.get('items', []) if isinstance(alignment, dict) else alignment
    duration = audio['duration_sec']
    provenance = {'source_audio': audio, 'config': config,
                  'audio_duration_sec': duration, 'timestamp_offset_sec': 0,
                  'processed_range_sec': [0.0, duration], 'generator_exhausted': True,
                  'ground_truth_in_model_inputs': False,
                  'context_sources': [], 'context': '',
                  'native_artifact': 'native_output.json',
                  'model': native.get('model', {}),
                  'timestamps_are': 'model_predictions_not_human_annotations'}
    if source_video:
        provenance['source_video'] = source_video
    return {'schema_version': '3.0', 'status': 'complete',
            'created_at': utc_now(), 'time_base': 'source_audio_seconds',
            'segments': native_segments(text, items, duration),
            'full_text': text, 'language': native.get('language'),
            'provenance': provenance, 'metrics': metrics or {},
            'coverage': {'processed_duration_sec': duration,
                         'processed_range_sec': [0.0, duration]},
            'accuracy_status': 'not_independently_evaluated'}
