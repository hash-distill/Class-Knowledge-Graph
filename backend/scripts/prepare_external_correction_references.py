"""Build scorer-only whole-clip references after independent ASR is frozen.

Never import or call this helper from a model input/generation script. It reads
official human reference text solely to prepare the evaluator's --reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_correction_models import file_sha256, read_json


def prepare_references(external_dir, evaluation_dir=None):
    external_dir = Path(external_dir).resolve()
    evaluation_dir = Path(evaluation_dir).resolve() if evaluation_dir else external_dir/'evaluation'
    manifest = read_json(external_dir/'manifest.json')
    inputs = read_json(external_dir/'model-inputs.json')
    gold = read_json(external_dir/'gold-references.json')
    if gold.get('use') != 'scorer_only_do_not_send_to_models':
        raise ValueError('Gold reference must be explicitly scorer-only')
    samples = {row['id']: row for row in manifest['samples']}
    input_rows = {row['id']: row for row in inputs['samples']}
    references = []
    if set(samples) != set(input_rows) or set(samples) != {row['id'] for row in gold['samples']}:
        raise ValueError('Manifest, model inputs and gold must contain exactly the same sample IDs')
    for row in gold['samples']:
        identifier = row['id']; sample = samples[identifier]; model_input = input_rows[identifier]
        if row.get('source') != 'official_manual_transcription':
            raise ValueError('Only official manual transcription is accepted by this adapter')
        if model_input.get('ground_truth_in_model_inputs') is not False:
            raise ValueError('Model-input separation must be explicitly recorded')
        wav = Path(model_input['wav_path'])
        audio_hash = file_sha256(wav)
        if audio_hash != sample['files']['wav']['sha256']:
            raise ValueError('Actual WAV differs from the frozen external manifest')
        annotation = Path(sample['files']['annotation']['path'])
        annotation_hash = file_sha256(annotation)
        if annotation_hash != row['source_annotation_sha256'] or annotation_hash != sample['files']['annotation']['sha256']:
            raise ValueError('Official annotation bytes differ from frozen reference provenance')
        paragraph = read_json(annotation).get('paragraphs', {})
        if not isinstance(paragraph, dict) or paragraph.get('content') != row.get('reference_text'):
            raise ValueError('Gold reference text differs from actual pinned official annotation content')
        recorded_interval = row.get('source_recording_interval_seconds')
        if recorded_interval is not None and recorded_interval != [paragraph.get('startTime'), paragraph.get('endTime')]:
            raise ValueError('Gold original interval differs from actual pinned official annotation')
        with wave.open(str(wav), 'rb') as stream:
            duration = stream.getnframes()/stream.getframerate()
        directory = evaluation_dir/identifier
        asr = read_json(directory/'asr.json')
        if asr.get('status') != 'complete':
            raise ValueError('Freeze complete independent ASR before creating scorer-only references')
        source_audio = asr.get('provenance', {}).get('source_audio', {})
        if source_audio.get('sha256') != audio_hash:
            raise ValueError('Independent ASR does not identify the actual source WAV')
        segments = asr.get('segments', [])
        if (len(segments) != 1 or segments[0].get('id') != 'T0001' or
                segments[0].get('start_sec') != 0 or
                abs(segments[0].get('end_sec', -1)-duration) > 1e-6):
            raise ValueError('Expected one whole-clip T0001 using actual WAV clock, without guessed word boundaries')
        reference = {'schema_version': '1.0', 'reference_kind': 'official_verbatim',
            'human_verified': True, 'independent_of_model_outputs': True, 'used_as_model_input': False,
            'time_base': 'source_audio_seconds', 'source_audio_sha256': audio_hash,
            'provenance': {'source': manifest.get('source_page'), 'dataset': manifest.get('dataset'),
                'source_revision': manifest.get('source_revision'), 'sample_id': identifier,
                'source_annotation_sha256': annotation_hash,
                'gold_references_sha256': file_sha256(external_dir/'gold-references.json'),
                'source_recording_interval_seconds': row.get('source_recording_interval_seconds'),
                'mapping': 'official whole-clip reference mapped to complete matching WAV; no word times inferred',
                'independence_notice': 'Official test annotations; model training overlap is unknown.'},
            'segments': [{'transcript_id': 'T0001', 'start_sec': 0., 'end_sec': duration,
                          'text': paragraph['content']}]}
        references.append((directory/'reference.json', reference))
    # Validate every sample first, then publish only scorer files.
    for path, reference in references:
        path.write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    terms_path = evaluation_dir/'empty-terms.json'
    terms_path.write_text(json.dumps({'kind': 'external_no_primary_lesson_vocabulary', 'groups': []}, indent=2)+'\n', encoding='utf-8')
    return {'reference_count': len(references), 'references': [str(path) for path, _ in references],
            'terminology': str(terms_path), 'scorer_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    args = parser.parse_args()
    try:
        result = prepare_references(args.external_dir, args.evaluation_dir)
    except (ValueError, KeyError, TypeError, OSError, wave.Error) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
