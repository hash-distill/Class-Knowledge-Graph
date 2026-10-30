"""Compare frozen correction runs without calling a model or changing a lesson.

An ASR hypothesis is not ground truth. This tool reports character changes,
proposal/guard provenance and vocabulary diagnostics. CER is enabled only by
an explicitly independent, verified, same-clock verbatim reference contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERMS = ROOT / 'tests/fixtures/cec4/chemistry-001-9-terminology.json'
MODEL_CHANNELS = {'model', 'llm', 'native_model', 'native', 'model_native'}
SOURCE_CHANNELS = {'source_vocabulary', 'secondary_asr', 'recurrent_context'}
CONDITION_KEYS = ('track', 'guard_method', 'system_sha256', 'prompt_sha256',
                  'temperature', 'seed', 'n_ctx', 'max_tokens', 'request_format')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def align_characters(before, after):
    """Exact Levenshtein alignment, per segment; never align across lesson gaps.

    Ties prefer substitution, deletion, then insertion. Unicode code points are
    the unit. Changed spans merge adjacent edit operations, preserving offsets
    on both strings. These spans are change counts, not correction counts.
    """
    n, m = len(before), len(after)
    distance = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i] + [0] * m
        for j in range(1, m + 1):
            row[j] = min(distance[i-1][j-1] + (before[i-1] != after[j-1]),
                         distance[i-1][j] + 1, row[j-1] + 1)
        distance.append(row)
    steps, i, j = [], n, m
    while i or j:
        if i and j and before[i-1] == after[j-1] and distance[i][j] == distance[i-1][j-1]:
            steps.append(('equal', i-1, i, j-1, j)); i -= 1; j -= 1
        elif i and j and distance[i][j] == distance[i-1][j-1] + 1:
            steps.append(('substitute', i-1, i, j-1, j)); i -= 1; j -= 1
        elif i and distance[i][j] == distance[i-1][j] + 1:
            steps.append(('delete', i-1, i, j, j)); i -= 1
        else:
            steps.append(('insert', i, i, j-1, j)); j -= 1
    steps.reverse()
    groups = []
    for op, i, end_i, j, end_j in steps:
        changed = op != 'equal'
        if groups and groups[-1]['changed'] == changed:
            groups[-1]['end_char'], groups[-1]['output_end_char'] = end_i, end_j
        else:
            groups.append({'changed': changed, 'start_char': i, 'end_char': end_i,
                           'output_start_char': j, 'output_end_char': end_j})
    spans = []
    for group in groups:
        if not group['changed']:
            continue
        item = {k: v for k, v in group.items() if k != 'changed'}
        item.update(original=before[item['start_char']:item['end_char']],
                    replacement=after[item['output_start_char']:item['output_end_char']])
        item['operation'] = ('insert' if not item['original'] else
                             'delete' if not item['replacement'] else 'replace')
        spans.append(item)
    counts = Counter(op for op, *_ in steps)
    return {'distance': distance[n][m], 'substitutions': counts['substitute'],
            'deletions': counts['delete'], 'insertions': counts['insert'],
            'equal_characters': counts['equal'], 'spans': spans, 'alignment': groups}


def protected_flags(before, after):
    patterns = {
        'arabic_number_sequence_changed': r'[0-9]+(?:[.．][0-9]+)?',
        'chinese_numeral_sequence_changed': r'[零〇一二三四五六七八九十百千万萬亿億两兩]+',
        'negation_character_sequence_changed': r'[不非无無未没沒莫勿别別否]',
        'formula_operator_sequence_changed': r'[=<>≤≥≠≈∑∫√×÷^_+*/−-]',
        'letter_or_formula_case_sequence_changed': r'[A-Za-z]+',
    }
    return [name for name, pattern in patterns.items()
            if re.findall(pattern, before) != re.findall(pattern, after)]


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def terminology_occurrences(text, terms):
    """Return literal known-form occurrences with raw code-point offsets."""
    found = []
    for term in terms:
        for form in term['forms']:
            pattern = r'\s*'.join(re.escape(c) for c in form)
            for match in re.finditer(pattern, text):
                found.append({'term_id': term['id'], 'form': match.group(),
                              'start_char': match.start(), 'end_char': match.end()})
    return found


def _touches(span, start, end):
    return (span['start_char'] < end and span['end_char'] > start or
            span['start_char'] == span['end_char'] and start < span['start_char'] < end)


def terminology_diagnostics(before, after, alignment, terms):
    before_occurrences = terminology_occurrences(before, terms)
    after_occurrences = terminology_occurrences(after, terms)
    touched, lost = [], []
    term_map = {term['id']: term for term in terms}
    for occurrence in before_occurrences:
        start, end = occurrence['start_char'], occurrence['end_char']
        if not any(_touches(span, start, end) for span in alignment['spans']):
            continue
        touched.append(occurrence)
        # Equal runs may cover more than this token; narrow their mapped bounds.
        positions = []
        for group in alignment['alignment']:
            if not _touches(group, start, end):
                continue
            a, b = group['output_start_char'], group['output_end_char']
            if not group['changed']:
                a += max(start, group['start_char']) - group['start_char']
                b -= group['end_char'] - min(end, group['end_char'])
            positions.append((a, b))
        mapped = after[min(a for a, _ in positions):max(b for _, b in positions)] if positions else ''
        if not any(compact(form) in compact(mapped) for form in term_map[occurrence['term_id']]['forms']):
            lost.append({**occurrence, 'mapped_output': mapped})
    # A term moved elsewhere in the same segment has not disappeared. Keep
    # local alignment uncertainty separately and count only net form loss.
    before_counts = Counter(row['term_id'] for row in before_occurrences)
    after_counts = Counter(row['term_id'] for row in after_occurrences)
    allowance = before_counts-after_counts
    net_lost, relocated = [], []
    for occurrence in lost:
        if allowance[occurrence['term_id']] > 0:
            net_lost.append(occurrence); allowance[occurrence['term_id']] -= 1
        else:
            relocated.append(occurrence)
    return {'present_group_ids': sorted({row['term_id'] for row in after_occurrences}),
            'literal_occurrence_count': len(after_occurrences),
            'preexisting_normative_form_occurrences_touched': touched,
            'preexisting_normative_form_occurrences_no_longer_normative': net_lost,
            'preexisting_normative_form_occurrences_relocated_or_locally_ambiguous': relocated}


def _source_segments(media_dir):
    path = media_dir / 'asr.json'
    source = read_json(path)
    if source.get('status') != 'complete':
        raise ValueError('Source ASR must have status=complete')
    rows = source['segments']
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Source transcript IDs must be present and unique')
    for row in rows:
        if not isinstance(row['text'], str) or not (0 <= row['start_sec'] < row['end_sec']):
            raise ValueError('Invalid source segment text or clock')
    provenance = source.get('provenance', {})
    return rows, {'source_asr_sha256': file_sha256(path),
                  'source_video_sha256': provenance.get('source_video', {}).get('sha256'),
                  'source_audio_sha256': provenance.get('source_audio', {}).get('sha256', provenance.get('audio_sha256')),
                  'audio_sha256': provenance.get('audio_sha256')}


def _assert_source_contract(result, source, source_hashes):
    identity_key = 'source_video_sha256' if source_hashes.get('source_video_sha256') else 'source_audio_sha256'
    if not source_hashes.get(identity_key):
        raise ValueError('Source requires a real source video or source audio hash')
    for key in ('source_asr_sha256', identity_key):
        if result.get(key) != source_hashes[key]:
            raise ValueError(f'{key} differs from the immutable lesson source')
    if result.get('segments') is not None:
        rows = result['segments']
        if len(rows) != len(source):
            raise ValueError('A comparison run must include every source segment')
        for original, row in zip(source, rows):
            if (row.get('transcript_id'), row.get('original_text'), row.get('start_sec'), row.get('end_sec')) != (
                    original['id'], original['text'], original['start_sec'], original['end_sec']):
                raise ValueError('Run segment ID/text/order/clock differs from immutable source')
            if not isinstance(row.get('text'), str):
                raise ValueError('Run corrected text must be a string')


def _model_proposals(result):
    ledger = result.get('proposal_ledger')
    if isinstance(ledger, list):
        proposals = [deepcopy(row) for row in ledger if row.get('channel') in MODEL_CHANNELS]
        calls = result.get('model_proposals')
        # A timed-out completion produced no recoverable raw model output.
        available = bool(proposals) or not calls or any(isinstance(call.get('answer'), dict) for call in calls)
        return proposals, available
    recorded = result.get('model_proposals')
    if not isinstance(recorded, list):
        return [], False
    proposals = []
    for call in recorded:
        answer = next((call[key] for key in ('answer', 'response', 'parsed_response', 'parsed_payload', 'payload')
                       if isinstance(call.get(key), dict)), None)
        if answer is None and isinstance(call.get('segments'), list):
            answer = call
        for row in (answer or {}).get('segments', []):
            if isinstance(row, dict) and isinstance(row.get('edits'), list):
                proposals.extend({'transcript_id': row.get('transcript_id'), 'channel': 'model',
                                  'candidate': edit, 'disposition': 'not_recorded'} for edit in row['edits'])
    return proposals, True


def apply_raw_proposals(source, proposals):
    """Apply locatable proposals before semantic guards; keep all failures visible.

    This is a diagnostic reconstruction, not a new model completion. It refuses
    ambiguous repeated substrings and conflicting offsets rather than guessing.
    """
    source_map = {row['id']: row for row in source}
    selected, failures, traces = {}, [], []
    for index, entry in enumerate(proposals):
        row = source_map.get(entry.get('transcript_id'))
        candidate = entry.get('candidate', {})
        trace = {**deepcopy(entry), 'proposal_index': index, 'potentially_unsafe_flags': []}
        before = candidate.get('original') if isinstance(candidate, dict) else None
        after = candidate.get('replacement') if isinstance(candidate, dict) else None
        failure = None
        if row is None or not isinstance(before, str) or not before or not isinstance(after, str):
            failure = 'invalid_or_missing_target_or_text'
        else:
            trace['potentially_unsafe_flags'] = protected_flags(before, after)
            start = candidate.get('original_start', candidate.get('start_char'))
            if isinstance(start, bool) or not isinstance(start, int):
                matches = [m.start() for m in re.finditer(re.escape(before), row['text'])]
                start = matches[0] if len(matches) == 1 else None
            if start is None:
                failure = 'ambiguous_or_missing_original_substring'
            elif start < 0 or row['text'][start:start+len(before)] != before:
                failure = 'offset_or_original_mismatch'
            else:
                end = start + len(before)
                existing = selected.setdefault(row['id'], [])
                if any(start < item['end_char'] and end > item['start_char'] for item in existing):
                    failure = 'overlapping_proposal'
                else:
                    existing.append({'start_char': start, 'end_char': end, 'replacement': after})
                    trace.update(start_char=start, end_char=end)
        trace['raw_application_status'] = failure or 'applied_without_semantic_guards'
        if failure:
            failures.append({'proposal_index': index, 'transcript_id': entry.get('transcript_id'), 'reason': failure})
        traces.append(trace)
    outputs = {}
    for row in source:
        text = row['text']
        for edit in sorted(selected.get(row['id'], []), key=lambda item: item['start_char'], reverse=True):
            text = text[:edit['start_char']] + edit['replacement'] + text[edit['end_char']:]
        outputs[row['id']] = text
    return outputs, failures, traces


def _native_outputs(envelope, result, source):
    native = envelope.get('native_outputs', result.get('native_outputs'))
    if native is None:
        return None
    if not isinstance(native, list) or len(native) != len(source):
        raise ValueError('Native output must cover all source segments')
    output = {}
    for original, row in zip(source, native):
        if row.get('transcript_id') != original['id'] or row.get('original_text') != original['text']:
            raise ValueError('Native output source/order differs from immutable source')
        if not isinstance(row.get('corrected_text'), str):
            raise ValueError('Native corrected_text must be a string')
        output[original['id']] = row['corrected_text']
    return output


def audit_runtime_samples(runtime):
    """Expose long measurement gaps such as suspend; never rewrite elapsed time."""
    runtime = deepcopy(runtime)
    paths = runtime.get('sample_files') or ([runtime['sample_file']] if runtime.get('sample_file') else [])
    gaps = []
    for path in paths:
        source = Path(path)
        if not source.is_file():
            continue
        samples = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
        samples = [row for row in samples if row.get('phase') == runtime.get('phase')]
        gaps.extend(b['monotonic_seconds']-a['monotonic_seconds'] for a, b in zip(samples, samples[1:]))
    if gaps:
        runtime['maximum_generation_sampling_gap_sec'] = max(gaps)
        runtime['timing_interruption_detected'] = max(gaps) > 5
        runtime['timing_valid_for_latency_comparison'] = max(gaps) <= 5
    return runtime


def normalize_reference_text(text, normalization):
    if normalization == 'strict':
        return text
    return ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def _reference_metrics(output, reference):
    result = {}
    for normalization in ('strict', 'no_whitespace_or_punctuation'):
        errors = denominator = 0
        parts = Counter()
        for identifier, expected in reference.items():
            expected = normalize_reference_text(expected, normalization)
            actual = normalize_reference_text(output[identifier], normalization)
            aligned = align_characters(expected, actual)
            errors += aligned['distance']; denominator += len(expected)
            for key in ('substitutions', 'deletions', 'insertions'):
                parts[key] += aligned[key]
        result[normalization] = {'character_errors': errors, 'reference_characters': denominator,
                                 'cer': errors / denominator if denominator else None, **parts}
    return result


def reference_character_transitions(source, output, reference, normalization='strict'):
    """Count repair/damage of aligned reference positions, not guessed edits.

    Substitute/deleted reference characters are wrong; equal characters are
    right. Insertions have no reference position and are counted separately.
    Repeated-text alignment can be non-unique, so these rates explicitly depend
    on the deterministic exact alignment documented by this evaluator.
    """
    counts = Counter()
    for row in source:
        expected = normalize_reference_text(reference[row['id']], normalization)
        before = align_characters(expected, normalize_reference_text(row['text'], normalization))
        after = align_characters(expected, normalize_reference_text(output[row['id']], normalization))
        def correct_positions(aligned):
            return {index for group in aligned['alignment'] if not group['changed']
                    for index in range(group['start_char'], group['end_char'])}
        old, new = correct_positions(before), correct_positions(after)
        all_positions = set(range(len(expected)))
        counts['reference_characters'] += len(expected)
        counts['initially_correct_reference_characters'] += len(old)
        counts['initially_incorrect_reference_characters'] += len(all_positions - old)
        counts['repaired_reference_characters'] += len(new - old)
        counts['damaged_previously_correct_reference_characters'] += len(old - new)
        counts['preserved_correct_reference_characters'] += len(old & new)
        counts['insertion_errors_before'] += before['insertions']
        counts['insertion_errors_after'] += after['insertions']
    result = dict(counts)
    result['repair_rate'] = counts['repaired_reference_characters'] / counts['initially_incorrect_reference_characters'] if counts['initially_incorrect_reference_characters'] else None
    result['damage_rate'] = counts['damaged_previously_correct_reference_characters'] / counts['initially_correct_reference_characters'] if counts['initially_correct_reference_characters'] else None
    result['unit'] = 'unicode_reference_character_positions_excluding_insertion_positions'
    result['normalization'] = normalization
    result['alignment_dependent'] = True
    return result


def evaluate_lane(source, output, terms, reference=None):
    traces, present = [], set()
    summary = Counter()
    for row in source:
        text = output[row['id']]
        alignment = align_characters(row['text'], text)
        diagnostics = terminology_diagnostics(row['text'], text, alignment, terms)
        flags = protected_flags(row['text'], text)
        for key in ('distance', 'substitutions', 'deletions', 'insertions', 'equal_characters'):
            summary[key] += alignment[key]
        summary['changed_segment_count'] += row['text'] != text
        summary['unreviewed_changed_span_count'] += len(alignment['spans'])
        summary['protected_pattern_flagged_segment_count'] += bool(flags)
        summary['preexisting_normative_form_occurrences_touched'] += len(diagnostics['preexisting_normative_form_occurrences_touched'])
        summary['preexisting_normative_form_occurrences_no_longer_normative'] += len(diagnostics['preexisting_normative_form_occurrences_no_longer_normative'])
        summary['source_characters'] += len(row['text']); summary['output_characters'] += len(text)
        present.update(diagnostics['present_group_ids'])
        traces.append({'transcript_id': row['id'], 'start_sec': row['start_sec'], 'end_sec': row['end_sec'],
                       'original_text': row['text'], 'text': text,
                       'character_edits': alignment['spans'], 'protected_pattern_flags': flags,
                       'terminology': diagnostics})
    # Counter omits zeros; the stable report contract includes every summary key.
    keys = ('distance', 'substitutions', 'deletions', 'insertions', 'equal_characters',
            'changed_segment_count', 'unreviewed_changed_span_count', 'source_characters',
            'output_characters', 'protected_pattern_flagged_segment_count',
            'preexisting_normative_form_occurrences_touched',
            'preexisting_normative_form_occurrences_no_longer_normative')
    return {'summary': {**{key: summary[key] for key in keys},
                        'terminology_group_presence_count': len(present),
                        'terminology_group_count': len(terms),
                        'present_terminology_group_ids': sorted(present)},
            'segments': traces, 'reference_metrics': _reference_metrics(output, reference) if reference else None,
            'reference_character_transitions': reference_character_transitions(source, output, reference) if reference else None,
            'normalized_reference_character_transitions': reference_character_transitions(source, output, reference, 'no_whitespace_or_punctuation') if reference else None}


def load_reference(path, source, hashes):
    reference = read_json(path)
    required = {'reference_kind': ('human_verbatim', 'official_verbatim'),
                'human_verified': (True,), 'independent_of_model_outputs': (True,),
                'used_as_model_input': (False,)}
    for key, allowed in required.items():
        if (reference.get(key) not in allowed or
                key in ('human_verified', 'independent_of_model_outputs', 'used_as_model_input') and type(reference.get(key)) is not bool):
            raise ValueError(f'Reference lacks independent verified verbatim contract: {key}')
    time_base = reference.get('time_base', reference.get('clock'))
    identity_key = ('source_audio_sha256' if time_base == 'source_audio_seconds' else
                    'source_video_sha256' if time_base == 'source_video_seconds' else None)
    if identity_key is None or not hashes.get(identity_key) or reference.get(identity_key) != hashes[identity_key]:
        raise ValueError('Reference source video/audio hash or time base differs from the same-clock source')
    if reference.get('clock') and reference['clock'] != time_base:
        raise ValueError('Reference clock and time_base declarations conflict')
    if not reference.get('provenance', {}).get('source'):
        raise ValueError('Reference requires an attributable provenance.source')
    rows = reference.get('segments', [])
    if len(rows) != len(source):
        raise ValueError('Reference must be explicitly aligned to every ASR source segment')
    output = {}
    for original, row in zip(source, rows):
        if (row.get('transcript_id'), row.get('start_sec'), row.get('end_sec')) != (
                original['id'], original['start_sec'], original['end_sec']) or not isinstance(row.get('text'), str):
            raise ValueError('Reference source IDs/order/clock differ; map the genuine reference explicitly')
        output[original['id']] = row['text']
    return output, {'path': str(Path(path).resolve()), 'sha256': file_sha256(path),
                    'reference_kind': reference['reference_kind'], 'provenance': reference['provenance'],
                    'time_base': time_base, identity_key: hashes[identity_key],
                    'contract_declarations_are_not_independent_authentication': True}


def evaluate_run(label, path, source, hashes, terms, reference=None):
    envelope = read_json(path)
    result = envelope.get('result', envelope)
    _assert_source_contract(result, source, hashes)
    proposals, available = _model_proposals(result)
    raw_text, application_failures, proposal_traces = apply_raw_proposals(source, proposals)
    native = _native_outputs(envelope, result, source)
    native_records = envelope.get('native_outputs', result.get('native_outputs', []))
    if native is not None:
        raw_text, available = native, True
    accepted = ({row['transcript_id']: row['text'] for row in result['segments']}
                if result.get('segments') is not None else None)
    conditions = deepcopy(envelope.get('conditions', {}))
    conditions.setdefault('track', envelope.get('track', 'controlled_json' if accepted else 'native_text'))
    conditions.setdefault('guard_method', result.get('method'))
    recorded_calls = result.get('model_proposals', [])
    for name, audit_key in (('system_sha256', 'system_digest'), ('prompt_sha256', 'prompt_digest')):
        digests = [call[audit_key] for call in recorded_calls if call.get(audit_key)]
        if digests:
            conditions.setdefault(name, digests[0] if len(digests) == 1 else digests)
    model_calls = result.get('provenance', {}).get('model_calls', [])
    ledger = result.get('proposal_ledger')
    origins = Counter(edit.get('origin', edit.get('channel', 'unrecorded'))
                      for row in result.get('segments', []) for edit in row.get('edits', []))
    dispositions = Counter(row.get('disposition', 'unrecorded') for row in proposals)
    rejected = result.get('rejected_edits', [])
    return {'label': label, 'path': str(Path(path).resolve()), 'sha256': file_sha256(path),
            'model': result.get('model', envelope.get('model', 'unknown')), 'status': result.get('status', 'unrecorded'),
            'conditions': conditions, 'runtime': audit_runtime_samples(envelope.get('runtime', {})),
            'native_failure_count': sum(row.get('status', 'complete') != 'complete' for row in native_records),
            'native_output_records': deepcopy(native_records),
            'model_calls': model_calls, 'raw_model_proposals_available': available,
            'raw_model_proposal_count': len(proposals) if available and native is None else None,
            'raw_model_output_kind': 'native_text' if native is not None else 'reconstructed_from_recorded_proposals',
            'raw_model_text_reconstruction_status': ('native_text' if native is not None else
                'incomplete_due_to_unapplied_proposals' if application_failures else 'complete' if available else 'unavailable'),
            'raw_application_failures': application_failures, 'model_proposal_traces': proposal_traces,
            'potentially_unsafe_model_proposal_count': sum(bool(row['potentially_unsafe_flags']) for row in proposal_traces) if proposals else (0 if available and native is None else None),
            'model_proposal_dispositions': dict(dispositions),
            'accepted_edit_origins': dict(origins), 'proposal_ledger_available': isinstance(ledger, list),
            'guard_rejected_proposal_count': result.get('rejected_edit_count', len(rejected)),
            'guard_rejection_reasons': dict(Counter(row.get('reason', 'unrecorded') for row in rejected)),
            'guard_rejections': rejected, 'proposal_ledger': ledger,
            'raw_model': evaluate_lane(source, raw_text, terms, reference) if available else None,
            'accepted': evaluate_lane(source, accepted, terms, reference) if accepted is not None else None,
            'uncertain_segment_count': result.get('uncertain_segment_count'),
            'human_verified': False}


def _span_keys(lane):
    return {(row['transcript_id'], edit['start_char'], edit['end_char'], edit['replacement'])
            for row in lane['segments'] for edit in row['character_edits']}


def compare_conditions(runs):
    comparison = []
    for left_index, left in enumerate(runs):
        for right in runs[left_index+1:]:
            unknown, different = [], []
            for key in CONDITION_KEYS:
                a, b = left['conditions'].get(key), right['conditions'].get(key)
                if a is None or b is None:
                    unknown.append(key)
                elif a != b:
                    different.append(key)
            comparison.append({'left': left['label'], 'right': right['label'],
                               'same_declared_conditions': not unknown and not different,
                               'unknown_conditions': unknown, 'different_conditions': different,
                               'interpretation': 'declared_conditions_match' if not unknown and not different else 'descriptive_comparison_only'})
    return comparison


def build_report(media_dir, runs, deterministic_path=None, terms_path=DEFAULT_TERMS, reference_path=None):
    media_dir = Path(media_dir).resolve()
    source, hashes = _source_segments(media_dir)
    term_fixture = read_json(terms_path)
    fixture_source = term_fixture.get('source_asr_sha256')
    if fixture_source and fixture_source != hashes['source_asr_sha256']:
        raise ValueError('Terminology fixture belongs to another ASR snapshot; provide --terminology')
    terms = term_fixture['groups']
    reference, reference_meta = load_reference(reference_path, source, hashes) if reference_path else (None, None)
    evaluated = [evaluate_run(label, path, source, hashes, terms, reference) for label, path in runs]
    if len({row['label'] for row in evaluated}) != len(evaluated):
        raise ValueError('Run labels must be unique')
    deterministic = (evaluate_run('deterministic-source-only', deterministic_path, source, hashes, terms, reference)
                     if deterministic_path else None)
    if deterministic:
        if deterministic['accepted'] is None:
            raise ValueError('Deterministic baseline requires accepted segments')
        if (deterministic['raw_model_proposal_count'] not in (0, None) or
                any(origin in MODEL_CHANNELS for origin in deterministic['accepted_edit_origins']) or
                not deterministic['raw_model_proposals_available'] and deterministic['conditions']['track'] != 'deterministic'):
            raise ValueError('Deterministic baseline must explicitly record source-only output with no model proposals')
        deterministic_keys = _span_keys(deterministic['accepted'])
        for run in evaluated:
            if run['accepted'] is None:
                continue
            keys = _span_keys(run['accepted'])
            run['deterministic_comparison'] = {
                'identical_final_text': [r['text'] for r in run['accepted']['segments']] == [r['text'] for r in deterministic['accepted']['segments']],
                'aligned_changed_spans_shared_with_deterministic': len(keys & deterministic_keys),
                'aligned_changed_spans_additional_to_deterministic': len(keys - deterministic_keys),
                'deterministic_spans_absent_from_final': len(deterministic_keys - keys),
                'notice': 'Additional/different aligned spans are not verified improvements. Shared changes cannot be attributed solely to the LLM.'}
    artifacts = {name: {'sha256': file_sha256(media_dir/name)}
                 for name in ('asr.json', 'asr.unguided.json', 'asr.secondary.json', 'slides.json', 'activity.json')
                 if (media_dir/name).is_file()}
    return {'schema_version': '1.0', 'generated_at': datetime.now(timezone.utc).isoformat(),
            'evaluation_kind': 'frozen_same_lesson_diagnostic', 'source': {**hashes, 'media_dir': str(media_dir),
            'segment_count': len(source), 'artifacts': artifacts, 'segment_text_digest': json_digest(source)},
            'reference': reference_meta, 'accuracy_evaluated': reference is not None,
            'mis_edit_rate': None, 'mis_edit_rate_status': 'requires_independent_changed_span_adjudication; reference-character damage is reported separately when available',
            'terminology': {'fixture_path': str(Path(terms_path).resolve()), 'fixture_sha256': file_sha256(terms_path),
                            'kind': term_fixture.get('kind'), 'groups': terms},
            'source_asr': evaluate_lane(source, {row['id']: row['text'] for row in source}, terms, reference),
            'runs': evaluated, 'deterministic_baseline': deterministic,
            'conditions_comparison': compare_conditions(evaluated),
            'limitations': [
                'Original ASR, unprompted ASR, the secondary recognizer, and projection OCR are not manual transcripts. Agreement among them is not independent evidence of accuracy.',
                'Term groups come from actual observations and machine hypotheses for this lesson. They check only occurrence and retention of standard forms and cannot be converted into terminology recall or accuracy.',
                'Unreviewed changes, potential protected-item changes, and guard rejections are counts, not actual erroneous correction rates. Numeric and negation character flags also cover homophones such as er fa and wei jing.',
                'Model candidates, guarded results, and source-rule results are reported separately. Shared source-rule revisions cannot all be attributed to model gains.',
                'When native correction and structured candidates use different prompts and task formats, comparisons describe system behavior only. Prompt and sampling conditions are unknown for historical results without request records.',
                'GPU memory and duration are copied only from observed runtime records. Distinguish idle usage, model loading, sampled peaks, and whole-workflow scope.',
                'Only one chemistry segment of about 90 seconds is used. The vocabulary fixed after observing this lesson has selection bias and cannot generalize to other lessons.']}


def markdown_report(report):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' / ')
    def value(value):
        return 'not recorded' if value is None else str(value)
    lines = ['# Same-lesson correction model comparison', '',
             f"Frozen input: {report['source']['segment_count']} segments; ASR SHA-256: `{report['source']['source_asr_sha256']}`.", '',
             'Independent verbatim reference: ' + ('provided and passed source and clock declaration checks; see JSON for CER.' if report['accuracy_evaluated'] else 'not provided. Correction rate, actual erroneous correction rate, and CER were not evaluated.'), '',
             '| Run | Output layer | Changed segments | Unreviewed change spans | Character edit distance | Term groups present | Existing standard forms lost | Segments with protected-item changes |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    all_runs = [{'label': 'Original ASR', 'accepted': report['source_asr'], 'raw_model': None},
                *([report['deterministic_baseline']] if report['deterministic_baseline'] else []), *report['runs']]
    for run in all_runs:
        for lane_name, lane in (('Native text / locatable candidate reconstruction', run['raw_model']), ('Guarded / rule results', run['accepted'])):
            if lane is None:
                continue
            summary = lane['summary']
            if run['label'] == 'Original ASR':
                lane_name = 'Original input'
            lines.append('| ' + ' | '.join(cell(x) for x in (run['label'], lane_name,
                summary['changed_segment_count'], summary['unreviewed_changed_span_count'], summary['distance'],
                f"{summary['terminology_group_presence_count']}/{summary['terminology_group_count']}",
                summary['preexisting_normative_form_occurrences_no_longer_normative'],
                summary['protected_pattern_flagged_segment_count'])) + ' |')
    if report['accuracy_evaluated']:
        lines += ['', '## Independent reference scoring', '',
                  'Repairs and damage are counted at reference-character positions; insertion errors are counted separately. The repair and damage counts below remove whitespace and punctuation and depend on optimal character alignment.', '',
                  '| Run | Output layer | Strict CER | CER without whitespace/punctuation | Repaired reference characters | Previously correct characters damaged | Repair rate | Damage rate | Insertion errors: before → after |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        def percentage(number):
            return 'no denominator' if number is None else f'{number:.2%}'
        for run in all_runs:
            for lane_name, lane in (('Raw model', run['raw_model']), ('Guarded / rules', run['accepted'])):
                if lane is None:
                    continue
                metrics = lane['reference_metrics']
                transitions = lane['normalized_reference_character_transitions']
                if run['label'] == 'Original ASR':
                    lane_name = 'Original input'
                lines.append('| ' + ' | '.join(cell(x) for x in (run['label'], lane_name,
                    percentage(metrics['strict']['cer']), percentage(metrics['no_whitespace_or_punctuation']['cer']),
                    transitions['repaired_reference_characters'], transitions['damaged_previously_correct_reference_characters'],
                    percentage(transitions['repair_rate']), percentage(transitions['damage_rate']),
                    f"{transitions['insertion_errors_before']}→{transitions['insertion_errors_after']}")) + ' |')
    lines += ['', '## Candidates, attribution, and runtime records', '']
    for run in report['runs']:
        lines += [f"### {run['label']}", '',
                  f"Model: `{run['model']}`; status: `{run['status']}`; task: `{run['conditions']['track']}`.", '',
                  f"Segments retaining the original text after a failed native request: {run['native_failure_count']}.", '',
                  f"Model proposals: {value(run['raw_model_proposal_count'])}; proposals with potential protected-item changes: {value(run['potentially_unsafe_model_proposal_count'])}; guard rejections (including source-rule candidates): {value(run['guard_rejected_proposal_count'])}.", '',
                  f"Model candidates that cannot be applied to the source: {len(run['raw_application_failures'])}; candidate reconstruction status: `{run['raw_model_text_reconstruction_status']}`.", '',
                  'Accepted edit origins: `' + json.dumps(run['accepted_edit_origins'], ensure_ascii=False) + '`. ', '',
                  'GPU memory and timing records: `' + json.dumps(run['runtime'], ensure_ascii=False) + '`. ', '']
        if not run['raw_model_proposals_available']:
            lines += ['Historical records did not retain raw model candidates, so raw model output cannot be recovered. Final changes cannot all be attributed to the model.', '']
        if run.get('deterministic_comparison'):
            comparison = run['deterministic_comparison']
            lines += [f"Final text identical to source rules alone: {comparison['identical_final_text']}; shared spans: {comparison['aligned_changed_spans_shared_with_deterministic']}; additional spans: {comparison['aligned_changed_spans_additional_to_deterministic']}. Additional spans still require audio verification.", '']
    lines += ['## Comparison conditions', '', '```json', json.dumps(report['conditions_comparison'], ensure_ascii=False, indent=2), '```', '',
              '## Changes by segment', '', '| Run | Output layer | Original segment and time | Source | Output | Protected-item flags |', '| --- | --- | --- | --- | --- | --- |']
    for run in report['runs']:
        for lane_name, lane in (('Raw model', run['raw_model']), ('Guarded', run['accepted'])):
            if lane is None:
                continue
            for row in lane['segments']:
                if row['character_edits']:
                    lines.append('| ' + ' | '.join(cell(x) for x in (run['label'], lane_name,
                        f"{row['transcript_id']} {row['start_sec']}–{row['end_sec']}s", row['original_text'], row['text'],
                        ', '.join(row['protected_pattern_flags']))) + ' |')
    lines += ['', '## Interpretation limits', ''] + [f'- {item}' for item in report['limitations']]
    return '\n'.join(lines) + '\n'


def write_report(report, output_dir):
    output_dir = Path(output_dir).resolve()
    media_dir = Path(report['source']['media_dir'])
    if output_dir == media_dir or media_dir in output_dir.parents:
        raise ValueError('Write comparison outputs outside the immutable lesson directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output_dir/'comparison.md').write_text(markdown_report(report), encoding='utf-8')
    fields = ('run', 'lane', 'transcript_id', 'start_sec', 'end_sec', 'original_text', 'text',
              'unreviewed_changed_span_count', 'protected_pattern_flags', 'character_edits')
    with (output_dir/'segments.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for run in report['runs']:
            for lane_name in ('raw_model', 'accepted'):
                if run[lane_name] is None:
                    continue
                for row in run[lane_name]['segments']:
                    writer.writerow({'run': run['label'], 'lane': lane_name,
                        **{key: row[key] for key in ('transcript_id', 'start_sec', 'end_sec', 'original_text', 'text')},
                        'unreviewed_changed_span_count': len(row['character_edits']),
                        'protected_pattern_flags': json.dumps(row['protected_pattern_flags'], ensure_ascii=False),
                        'character_edits': json.dumps(row['character_edits'], ensure_ascii=False)})
    with (output_dir/'proposals.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ('run', 'transcript_id', 'channel', 'disposition', 'reason', 'candidate', 'potentially_unsafe_flags', 'raw_application_status')
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for run in report['runs']:
            for row in run['model_proposal_traces']:
                writer.writerow({'run': run['label'], **{key: json.dumps(row.get(key), ensure_ascii=False)
                                if key in ('candidate', 'potentially_unsafe_flags') else row.get(key, '') for key in fields[1:]}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=JSON')
    parser.add_argument('--deterministic-run', type=Path)
    parser.add_argument('--terminology', type=Path, default=DEFAULT_TERMS)
    parser.add_argument('--reference', type=Path, help='Independent verified same-clock verbatim reference, never secondary ASR')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    runs = []
    for item in args.run:
        if '=' not in item or not item.split('=', 1)[0]:
            parser.error('--run requires LABEL=JSON')
        label, path = item.split('=', 1); runs.append((label, Path(path)))
    try:
        report = build_report(args.media_dir, runs, args.deterministic_run, args.terminology, args.reference)
        write_report(report, args.output_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'status': 'complete', 'output_dir': str(args.output_dir.resolve()),
                      'accuracy_evaluated': report['accuracy_evaluated'], 'runs': [row['label'] for row in report['runs']]}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
