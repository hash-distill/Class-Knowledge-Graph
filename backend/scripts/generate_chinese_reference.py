"""Recognize the prepared lesson WAV on the configured A6000, preserving ASR.

The remote environment is separate from the review API. Its model cache is
reused; each invocation uses a unique job directory. No server paths or audio
are inferred from another lesson, and only verified JSON is published locally.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.common import read_json, sha256


def run(*arguments):
    subprocess.run(list(arguments), check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--media-dir', required=True, type=Path)
    p.add_argument('--host', default=os.getenv('REVIEW_ASR_SSH_HOST'), help='Explicit SSH alias, or REVIEW_ASR_SSH_HOST')
    p.add_argument('--remote-python', default=os.getenv('REVIEW_ASR_REMOTE_PYTHON','python'))
    p.add_argument('--remote-root', default=os.getenv('REVIEW_ASR_REMOTE_ROOT','/tmp/jingxi-asr'))
    p.add_argument('--gpu', type=int, default=os.getenv('REVIEW_ASR_GPU','0'))
    args = p.parse_args()
    if not args.host or args.host.startswith('-'):
        p.error('Specify --host or REVIEW_ASR_SSH_HOST before sending audio to an explicitly configured SSH server')
    if args.gpu < 0:p.error('--gpu must be nonnegative')
    directory = args.media_dir.resolve()
    primary = read_json(directory / 'asr.json')
    if primary.get('status') != 'complete':
        p.error('Prepare a complete source ASR artifact before the independent reference')
    video_sha = primary['provenance']['source_video']['sha256']
    audio = directory / 'audio.wav'
    audio_sha = sha256(audio)
    if primary['provenance'].get('audio_sha256') not in (None, audio_sha):
        p.error('Prepared WAV differs from the original ASR audio')
    job = args.remote_root.rstrip('/') + '/jobs/' + video_sha[:12] + '-' + uuid.uuid4().hex[:12]
    remote_script = job + '/recognize_secondary_asr.py'
    remote_audio, remote_output = job + '/audio.wav', job + '/asr.secondary.json'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.host]
    run(*ssh, shlex.join(['mkdir', '-p', job + '/tmp']))
    run('scp', '-o', 'BatchMode=yes', str(Path(__file__).with_name('recognize_secondary_asr.py')), args.host + ':' + remote_script)
    run('scp', '-o', 'BatchMode=yes', str(audio), args.host + ':' + remote_audio)
    command = ['env', 'CUDA_VISIBLE_DEVICES=' + str(args.gpu), 'OMP_NUM_THREADS=4',
               'MODELSCOPE_CACHE=' + args.remote_root.rstrip('/') + '/models', 'TMPDIR=' + job + '/tmp',
               args.remote_python, '-u', remote_script, '--audio', remote_audio,
               '--source-sha256', video_sha, '--output', remote_output]
    run(*ssh, shlex.join(command))
    temporary = directory / ('asr.secondary.' + uuid.uuid4().hex + '.download.json')
    run('scp', '-o', 'BatchMode=yes', args.host + ':' + remote_output, str(temporary))
    reference = read_json(temporary)
    provenance = reference.get('provenance', {})
    if (reference.get('status') != 'complete' or not reference.get('segments')
            or provenance.get('source_video', {}).get('sha256') != video_sha
            or provenance.get('audio_sha256') != audio_sha or sha256(audio) != audio_sha):
        raise ValueError('Independent recognition did not match the same original audio/video')
    destination = directory / 'asr.secondary.json'
    temporary.replace(destination)
    print(json.dumps({'output': str(destination), 'segments': len(reference['segments']),
                      'source_sha256': video_sha, 'audio_sha256': audio_sha}, ensure_ascii=False))


if __name__ == '__main__':
    main()
