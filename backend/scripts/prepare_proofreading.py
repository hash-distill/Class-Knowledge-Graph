"""Freeze correction evidence independently of graph/report publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review.common import read_json, sha256, validate_segments
from review.correction_provider import CorrectionProvider
from review.mutation import LessonMutationLock
from review.proofreading import load_or_proofread
from review.timebase import validate_media_timebase


def prepare(media_dir,video,*,provider,output=None,force=False):
    directory,video=Path(media_dir).resolve(),Path(video).resolve()
    with LessonMutationLock(directory):
        asr,pages=read_json(directory/'asr.json'),read_json(directory/'slides.json')
        video_hash=sha256(video)
        for artifact in (asr,pages):
            if artifact.get('status') != 'complete':
                raise ValueError('Media extraction is incomplete')
            if artifact.get('provenance',{}).get('source_video',{}).get('sha256') != video_hash:
                raise ValueError('ASR/slides/video belong to different sources')
        duration=validate_media_timebase(asr,pages)
        transcript=validate_segments(asr['segments'],duration,'transcript')
        slides=validate_segments(pages['slides'],duration,'slide')
        if not transcript:
            raise ValueError('No speech transcript was produced')
        if set(row['id'] for row in transcript) & set(row['id'] for row in slides):
            raise ValueError('Evidence IDs must be globally unique')
        try:
            return load_or_proofread(directory,transcript,slides,provider,video_hash,
                artifact_path=output,force=force)
        finally:
            if hasattr(provider,'close'):
                provider.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir',required=True,type=Path)
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--config',type=Path,help='Dedicated correction model profile')
    parser.add_argument('--output',type=Path,help='Separate comparison artifact; omitting publishes only transcript.proofreading.json')
    parser.add_argument('--force',action='store_true',help='Recompute even when the audited cache matches')
    parser.add_argument('--no-managed-lifecycle',action='store_true',help='Caller owns model start/stop, e.g. measured comparisons')
    args=parser.parse_args()
    provider=CorrectionProvider(config_path=args.config,
        managed_lifecycle=False if args.no_managed_lifecycle else None)
    result=prepare(args.media_dir,args.video,provider=provider,output=args.output,force=args.force)
    print(json.dumps({'status':result['status'],'model':result['model'],
        'output':str((args.output or args.media_dir/'transcript.proofreading.json').resolve()),
        'edit_count':result['edit_count'],'contribution_counts':result['contribution_counts'],
        'source_asr_sha256':result['source_asr_sha256'],
        'source_video_sha256':result['source_video_sha256']},ensure_ascii=False))
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
