"""Offline comparisons preserve provenance and never manufacture error labels."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_correction_models import (DEFAULT_TERMS, align_characters, apply_raw_proposals,
    build_report, compare_conditions, evaluate_lane, file_sha256, load_reference,
    reference_character_transitions, write_report)
from scripts.prepare_external_correction_references import prepare_references


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def source_fixture(tmp_path, text='知道未经吗和谷氨酸钠'):
    media = tmp_path/'media'; media.mkdir()
    rows = [{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': text}]
    save(media/'asr.json', {'status': 'complete', 'segments': rows,
         'provenance': {'source_video': {'sha256': 'b'*64}, 'audio_sha256': 'c'*64}})
    terms = save(tmp_path/'terms.json', {'kind': 'test_only', 'groups': [
        {'id': 'msg', 'forms': ['味精']}, {'id': 'glutamate', 'forms': ['谷氨酸钠']}]})
    result = {'status': 'complete', 'method': 'test_guard', 'model': 'test_only',
              'source_asr_sha256': file_sha256(media/'asr.json'), 'source_video_sha256': 'b'*64,
              'human_verified': False, 'segments': [{'id': 'P0001', 'transcript_id': 'T0001',
              'start_sec': 0., 'end_sec': 4., 'original_text': text, 'text': text,
              'edits': [], 'uncertainties': []}], 'rejected_edit_count': 0}
    return media, rows, terms, result


@pytest.mark.parametrize('before,after,distance', [
    ('甲乙丙', '甲改丙', 1), ('甲乙', '甲新乙', 1), ('甲乙丙', '甲丙', 1),
    ('', '氨基酸', 3), ('氨基酸', '', 3), ('未修改', '未修改', 0),
    ('11.2', '1.12', 2), ('二发氨基酸', '阿尔法氨基酸', 3)])
def test_exact_character_alignment_reconstructs_output(before, after, distance):
    aligned = align_characters(before, after)
    rebuilt = before
    for span in reversed(aligned['spans']):
        assert before[span['start_char']:span['end_char']] == span['original']
        rebuilt = rebuilt[:span['start_char']] + span['replacement'] + rebuilt[span['end_char']:]
    assert rebuilt == after
    assert aligned['distance'] == distance
    assert sum(aligned[key] for key in ('substitutions', 'deletions', 'insertions')) == distance


def test_historical_final_output_is_not_claimed_as_model_candidate_or_accuracy(tmp_path):
    media, _, terms, result = source_fixture(tmp_path)
    result['segments'][0]['text'] = '知道味精吗和谷氨酸钠'
    path = save(tmp_path/'historical.json', result)
    report = build_report(media, [('historical', path)], terms_path=terms)
    run = report['runs'][0]
    assert report['accuracy_evaluated'] is False and report['mis_edit_rate'] is None
    assert run['raw_model'] is None and run['raw_model_proposal_count'] is None
    assert run['accepted']['summary']['unreviewed_changed_span_count'] == 1
    assert run['accepted']['reference_metrics'] is None
    assert run['runtime'] == {}
    write_report(report, tmp_path/'report')
    assert all((tmp_path/'report'/name).is_file() for name in ('comparison.json', 'comparison.md', 'segments.csv', 'proposals.csv'))
    assert '未提供' in (tmp_path/'report/comparison.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='outside'):
        write_report(report, media/'report')


def test_raw_model_guarded_final_and_source_only_contributions_are_separate(tmp_path):
    media, _, terms, result = source_fixture(tmp_path)
    source_edit = {'original': '未经', 'replacement': '味精', 'origin': 'source_vocabulary'}
    result['segments'][0].update(text='知道味精吗和谷氨酸钠', edits=[source_edit])
    deterministic = deepcopy(result)
    deterministic.update(proposal_ledger=[], model_proposals=[])
    model_candidate = {'original': '谷氨酸钠', 'replacement': '谷氨酸纳', 'evidence_ids': ['T0001']}
    result['proposal_ledger'] = [{'transcript_id': 'T0001', 'channel': 'model', 'candidate': model_candidate,
                                'disposition': 'rejected', 'reason': 'test_only'}]
    result['rejected_edits'] = [{'transcript_id': 'T0001', 'candidate': model_candidate, 'reason': 'test_only'}]
    result['rejected_edit_count'] = 1
    run_path, deterministic_path = save(tmp_path/'model.json', result), save(tmp_path/'source.json', deterministic)
    report = build_report(media, [('model', run_path)], deterministic_path, terms)
    run = report['runs'][0]
    assert run['raw_model']['segments'][0]['text'] == '知道未经吗和谷氨酸纳'
    assert run['accepted']['segments'][0]['text'] == '知道味精吗和谷氨酸钠'
    assert run['accepted_edit_origins'] == {'source_vocabulary': 1}
    assert run['guard_rejected_proposal_count'] == 1
    assert run['deterministic_comparison']['identical_final_text'] is True
    assert run['deterministic_comparison']['aligned_changed_spans_additional_to_deterministic'] == 0
    assert run['raw_model']['summary']['preexisting_normative_form_occurrences_no_longer_normative'] == 1


def test_raw_proposals_do_not_guess_repeated_spans_or_hide_conflicts():
    rows = [{'id': 'T1', 'text': '错词和错词'}]
    proposals = [
        {'transcript_id': 'T1', 'candidate': {'original': '错词', 'replacement': '正词'}},
        {'transcript_id': 'T1', 'candidate': {'original': '错词', 'replacement': '正词', 'original_start': 0}},
        {'transcript_id': 'T1', 'candidate': {'original': '错词', 'replacement': '别词', 'original_start': 0}}]
    outputs, failures, traces = apply_raw_proposals(rows, proposals)
    assert outputs == {'T1': '正词和错词'}
    assert [row['reason'] for row in failures] == ['ambiguous_or_missing_original_substring', 'overlapping_proposal']
    assert len(traces) == 3


def test_numeric_negation_flags_are_review_signals_even_when_correct_repair():
    source = [{'id': 'T1', 'text': '二发氨基酸知道未经吗', 'start_sec': 0., 'end_sec': 4.}]
    lane = evaluate_lane(source, {'T1': '阿尔法氨基酸知道味精吗'}, [])
    assert lane['summary']['protected_pattern_flagged_segment_count'] == 1
    assert lane['reference_metrics'] is None
    assert lane['segments'][0]['protected_pattern_flags'] == ['chinese_numeral_sequence_changed', 'negation_character_sequence_changed']


def test_relocated_normative_term_is_not_reported_as_lost():
    source = [{'id': 'T1', 'text': '谷氨酸咱们同学有所涉及', 'start_sec': 0., 'end_sec': 4.}]
    lane = evaluate_lane(source, {'T1': '咱们同学有所涉及谷氨酸'}, [{'id': 'glutamate', 'forms': ['谷氨酸']}])
    assert lane['summary']['preexisting_normative_form_occurrences_no_longer_normative'] == 0


def test_reference_character_repair_damage_and_insertions_are_explicit():
    source = [{'id': 'T1', 'text': '甲错丙'}]
    transitions = reference_character_transitions(source, {'T1': '改乙丙多'}, {'T1': '甲乙丙'})
    assert transitions['repaired_reference_characters'] == 1
    assert transitions['damaged_previously_correct_reference_characters'] == 1
    assert transitions['repair_rate'] == 1
    assert transitions['damage_rate'] == .5
    assert transitions['insertion_errors_before'] == 0
    assert transitions['insertion_errors_after'] == 1


def test_machine_reference_or_wrong_clock_cannot_enable_cer(tmp_path):
    media, source, _, result = source_fixture(tmp_path)
    reference = {'reference_kind': 'human_verbatim', 'human_verified': True,
        'independent_of_model_outputs': True, 'used_as_model_input': False,
        'clock': 'source_video_seconds', 'source_video_sha256': 'b'*64,
        'provenance': {'source': 'test-only synthetic fixture, not real annotation'},
        'segments': [{'transcript_id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '知道味精吗和谷氨酸钠'}]}
    hashes = {'source_video_sha256': result['source_video_sha256']}
    for mutation in ({'reference_kind': 'independent_machine_asr_unreviewed'},
                     {'human_verified': False}, {'used_as_model_input': True},
                     {'source_video_sha256': 'd'*64}):
        path = save(tmp_path/'reference.json', {**reference, **mutation})
        with pytest.raises(ValueError):
            load_reference(path, source, hashes)
    bad_clock = deepcopy(reference); bad_clock['segments'][0]['end_sec'] = 3.9
    with pytest.raises(ValueError, match='clock'):
        load_reference(save(tmp_path/'reference.json', bad_clock), source, hashes)
    outputs, meta = load_reference(save(tmp_path/'reference.json', reference), source, hashes)
    lane = evaluate_lane(source, outputs, [], outputs)
    assert lane['reference_metrics']['strict']['cer'] == 0
    assert lane['reference_character_transitions']['repaired_reference_characters'] == 2
    assert meta['contract_declarations_are_not_independent_authentication'] is True


def test_source_hash_text_and_clock_mismatches_are_rejected(tmp_path):
    media, _, terms, result = source_fixture(tmp_path)
    for mutation in ('hash', 'text', 'clock', 'order'):
        changed = deepcopy(result)
        if mutation == 'hash': changed['source_asr_sha256'] = 'd'*64
        if mutation == 'text': changed['segments'][0]['original_text'] += '伪造'
        if mutation == 'clock': changed['segments'][0]['start_sec'] = 0.1
        if mutation == 'order': changed['segments'][0]['transcript_id'] = 'T9999'
        with pytest.raises(ValueError):
            build_report(media, [('invalid', save(tmp_path/'invalid.json', changed))], terms_path=terms)


def test_native_text_is_reported_before_guards_and_never_implied_as_accepted(tmp_path):
    media, _, terms, result = source_fixture(tmp_path)
    original = result.pop('segments')[0]['original_text']
    envelope = {'track': 'native_text', 'result': result, 'native_outputs': [
        {'transcript_id': 'T0001', 'original_text': original, 'corrected_text': '知道味精吗和谷氨酸钠'}],
        'runtime': {'peak_device_used_mib': 4000, 'scope': 'test-only synthetic measurement'}}
    run = build_report(media, [('native', save(tmp_path/'native.json', envelope))], terms_path=terms)['runs'][0]
    assert run['accepted'] is None
    assert run['raw_model']['summary']['changed_segment_count'] == 1
    assert run['raw_model_proposal_count'] is None
    assert run['runtime']['scope'] == 'test-only synthetic measurement'


def test_native_and_json_with_missing_settings_are_not_called_controlled_pair():
    runs = [{'label': 'A', 'conditions': {'track': 'native_text'}},
            {'label': 'B', 'conditions': {'track': 'controlled_json'}}]
    pair = compare_conditions(runs)[0]
    assert pair['same_declared_conditions'] is False
    assert pair['different_conditions'] == ['track']
    assert 'temperature' in pair['unknown_conditions']


def test_model_proposal_payloads_are_audited_when_ledger_is_not_available(tmp_path):
    media, _, terms, result = source_fixture(tmp_path)
    result['model_proposals'] = [{'system_digest': 'system', 'prompt_digest': 'prompt',
        'answer': {'segments': [{'transcript_id': 'T0001', 'edits': [{'original': '未经', 'replacement': '味精'}]}]}}]
    run = build_report(media, [('A', save(tmp_path/'model.json', result))], terms_path=terms)['runs'][0]
    assert run['raw_model_proposal_count'] == 1
    assert run['conditions']['system_sha256'] == 'system'
    assert run['raw_model']['segments'][0]['text'] == '知道味精吗和谷氨酸钠'


def test_deterministic_baseline_cannot_contain_model_proposals(tmp_path):
    media, _, terms, result = source_fixture(tmp_path)
    result['model_proposals'] = [{'answer': {'segments': [{'transcript_id': 'T0001', 'edits': [
        {'original': '未经', 'replacement': '味精'}]}]}}]
    path = save(tmp_path/'not-deterministic.json', result)
    with pytest.raises(ValueError, match='source-only'):
        build_report(media, [('A', path)], path, terms)


def test_frozen_fourteen_group_fixture_is_diagnostic_not_error_annotation():
    fixture = json.loads(DEFAULT_TERMS.read_text(encoding='utf-8'))
    assert len(fixture['groups']) == 14
    assert fixture['kind'] == 'post_observation_normative_vocabulary_diagnostic_not_verbatim_ground_truth'
    assert len({row['id'] for row in fixture['groups']}) == 14
    assert all(row['evidence'] for row in fixture['groups'])


def test_real_audio_clock_is_supported_without_inventing_a_source_video(tmp_path):
    media, rows, terms, result = source_fixture(tmp_path)
    asr = json.loads((media/'asr.json').read_text(encoding='utf-8'))
    asr['provenance'] = {'source_audio': {'sha256': 'c'*64, 'duration_sec': 4.}}
    save(media/'asr.json', asr)
    result.pop('source_video_sha256')
    result['source_audio_sha256'] = 'c'*64
    result['source_asr_sha256'] = file_sha256(media/'asr.json')
    path = save(tmp_path/'audio-run.json', result)
    reference = {'reference_kind': 'official_verbatim', 'human_verified': True,
        'independent_of_model_outputs': True, 'used_as_model_input': False,
        'time_base': 'source_audio_seconds', 'source_audio_sha256': 'c'*64,
        'provenance': {'source': 'test-only synthetic official reference fixture'},
        'segments': [{'transcript_id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': rows[0]['text']}]}
    reference_path = save(tmp_path/'reference.json', reference)
    report = build_report(media, [('audio', path)], terms_path=terms, reference_path=reference_path)
    assert report['source']['source_video_sha256'] is None
    assert report['source']['source_audio_sha256'] == 'c'*64
    assert report['accuracy_evaluated'] is True
    assert report['reference']['time_base'] == 'source_audio_seconds'
    assert report['runs'][0]['accepted']['reference_metrics']['strict']['cer'] == 0


def test_reference_helper_maps_whole_wave_only_after_frozen_independent_asr(tmp_path):
    directory = tmp_path/'external'; directory.mkdir()
    audio = directory/'test-only.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(b'\x00\x00'*1600)
    annotation = save(directory/'annotation.json', {'paragraphs': {'content': '测试专用合成参考'}})
    save(directory/'manifest.json', {'dataset': 'test-only synthetic', 'source_page': 'https://example.invalid/test-only',
        'samples': [{'id': 'sample1', 'files': {'wav': {'sha256': file_sha256(audio)},
                    'annotation': {'path': str(annotation), 'sha256': file_sha256(annotation)}}}]})
    save(directory/'model-inputs.json', {'samples': [{'id': 'sample1', 'wav_path': str(audio),
         'ground_truth_in_model_inputs': False}]})
    save(directory/'gold-references.json', {'use': 'scorer_only_do_not_send_to_models',
         'samples': [{'id': 'sample1', 'source': 'official_manual_transcription',
                      'source_annotation_sha256': file_sha256(annotation), 'reference_text': '测试专用合成参考'}]})
    evaluation = directory/'evaluation/sample1'; evaluation.mkdir(parents=True)
    asr_path = evaluation/'asr.json'
    source = {'status': 'complete', 'provenance': {'source_audio': {'sha256': file_sha256(audio)}},
         'segments': [{'id': 'T0001', 'start_sec': 0., 'end_sec': .1, 'text': '独立草稿'}]}
    save(asr_path, source)
    frozen_hash = file_sha256(asr_path)
    result = prepare_references(directory)
    reference = json.loads((evaluation/'reference.json').read_text(encoding='utf-8'))
    assert result['scorer_only'] is True and result['reference_count'] == 1
    assert reference['segments'][0]['end_sec'] == .1
    assert reference['segments'][0]['text'] == '测试专用合成参考'
    assert file_sha256(asr_path) == frozen_hash
    source['segments'][0]['end_sec'] = .09; save(asr_path, source)
    with pytest.raises(ValueError, match='whole-clip'):
        prepare_references(directory)
