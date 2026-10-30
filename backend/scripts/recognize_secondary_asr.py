"""Generate an independent Mandarin ASR reference without replacing native ASR.

Run in a FunASR environment (optionally on the A6000 server). The output belongs
to the supplied source-video SHA and exact WAV bytes and is still machine text.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', default='paraformer-zh')
    parser.add_argument('--revision', default='v2.0.4')
    args = parser.parse_args()
    if len(args.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.source_sha256):
        parser.error('--source-sha256 must be the original video SHA-256')
    from funasr import AutoModel

    start = time.perf_counter()
    model = AutoModel(model=args.model, model_revision=args.revision,
                      vad_model='fsmn-vad', vad_model_revision='v2.0.4',
                      punc_model='ct-punc', punc_model_revision='v2.0.4',
                      vad_kwargs={'max_single_segment_time': 30000},
                      device=args.device, ncpu=4, disable_update=True,
                      trust_remote_code=False)
    # Intentionally no lecture-specific prompt/hotwords: an independent reference
    # should not merely echo terms supplied by the other recognizer or courseware.
    result = model.generate(input=str(args.audio), batch_size_s=90,
                            sentence_timestamp=True, return_raw_text=True)
    elapsed = time.perf_counter() - start
    sentences = [row for item in result for row in item.get('sentence_info', [])]
    if not sentences:
        raise RuntimeError('Recognizer did not return sentence timestamps; refusing to invent alignment')
    rows = []
    for row in sentences:
        text = row.get('text', '').strip()
        left, right = float(row['start']) / 1000, float(row['end']) / 1000
        if not text or left < 0 or right <= left:
            raise ValueError('Invalid timestamped secondary ASR sentence')
        rows.append({'id': f'R{len(rows)+1:04d}', 'start_sec': left,
                     'end_sec': right, 'text': text,
                     'quality': {'status': 'independent_machine_asr_unreviewed'}})
    model_root = Path(model.model_path)
    model_files = {str(p.relative_to(model_root)): digest(p)
                   for p in model_root.iterdir() if p.is_file() and p.suffix in ('.yaml', '.json', '.pb', '.pt', '.bin')}
    payload = {'schema_version': '1.0', 'status': 'complete', 'segments': rows,
               'raw_result': result,
               'provenance': {'method': 'independent_mandarin_asr',
                   'source_video': {'sha256': args.source_sha256},
                   'audio_sha256': digest(args.audio), 'model': args.model,
                   'model_revision': args.revision, 'resolved_model': str(model_root),
                   'model_file_sha256': model_files,
                   'funasr_version': importlib.metadata.version('funasr'),
                   'device': args.device, 'hotwords': [], 'human_verified': False,
                   'elapsed_seconds_including_load': round(elapsed, 3),
                   'generated_at': datetime.now(timezone.utc).isoformat()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'segments': len(rows),
                      'elapsed_seconds_including_load': round(elapsed, 3)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
