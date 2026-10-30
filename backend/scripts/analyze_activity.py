"""Analyze actual same-source classroom video; never manufacture missing behavior."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.activity import analyze_video, enrich_activity, attach_activity, validate_activity
from review.common import read_json, atomic_json
from review.mutation import LessonMutationLock


def main():
    parser = argparse.ArgumentParser(description='Full-video sampled behavior observations and trained classroom context')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--lesson-id', required=True)
    parser.add_argument('--media-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--context-model', type=Path)
    parser.add_argument('--pose-config', type=Path)
    parser.add_argument('--aric-report', type=Path)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--scene-threshold', type=float, default=.22)
    parser.add_argument('--enrich-only', action='store_true', help='Attach trained context/offline metrics to existing complete observations without detector rerun')
    args = parser.parse_args()
    if args.enrich_only:
        result = enrich_activity(read_json(args.media_dir/'activity.json'), args.video,
                                 context_model=args.context_model, aric_report=args.aric_report, device=args.device)
        validate_activity(result, {'lesson': {'id': args.lesson_id, 'duration_sec': result['duration_sec'],
                                             'source_video': {'sha256': result['source_sha256']}}}, args.media_dir)
        if (args.media_dir/'bundle.json').is_file(): attach_activity(args.media_dir, result)
        else:
            with LessonMutationLock(args.media_dir): atomic_json(args.media_dir/'activity.json', result)
    else:
        result = analyze_video(args.video, args.lesson_id, args.media_dir, interval_sec=args.interval,
            device=args.device, context_model=args.context_model, pose_config=args.pose_config,
            aric_report=args.aric_report, imgsz=args.imgsz, scene_threshold=args.scene_threshold)
    print('Saved complete activity:', args.media_dir/'activity.json')
    print('Summary:', result['summary'])


if __name__ == '__main__': main()
