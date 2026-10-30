"""Freeze only WAV inputs and identity; never read scorer/reference artifacts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.qwen_asr import atomic_json, inspect_wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root')
    a = p.parse_args()
    primary = a.root / 'artifacts/unified-chemistry-001-9'
    audio = inspect_wav(primary / 'audio.wav', 'c50ef88f904d383506b0da940e1e84878d4377fe71898015dd8aa972d8c822dc')
    provenance = json.loads((primary / 'audio.provenance.json').read_text(encoding='utf-8-sig'))
    samples = [{'id': primary.name, 'wav_path': str(primary / 'audio.wav'),
                'original_wav_path': str(primary / 'audio.wav'), 'sha256': audio['sha256'],
                'source_video': provenance['provenance']['source_video']}]
    external = a.root / 'artifacts/verification/cec4-comparison/external-chinese-lips/model-inputs.json'
    for row in json.loads(external.read_text(encoding='utf-8-sig'))['samples']:
        audio = inspect_wav(row['wav_path'])
        samples.append({'id': row['id'], 'wav_path': row['wav_path'],
                        'original_wav_path': row['wav_path'], 'sha256': audio['sha256']})
    if a.remote_root:
        for row in samples:
            row['wav_path'] = a.remote_root.rstrip('/') + '/inputs/' + row['id'] + '.wav'
    atomic_json(a.output, {'samples': samples, 'ground_truth_in_model_inputs': False})


if __name__ == '__main__':
    main()
