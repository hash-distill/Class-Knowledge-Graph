"""Video runner with complete JSON arrays, explicit context and run provenance."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import cv2
import yaml
from src.pretrained_pipeline import PretrainedClassroomPipeline
from src.snapshot_writer import SnapshotWriter


def run_pretrained(args):
    from scripts.infer_video import extract_anchor_events, open_capture, parse_crop_box, sampled_frames, video_timestamp
    source = args.source or args.student_source
    roi = [float(v) for v in args.student_roi.split(',')] if args.student_roi else None
    crop = parse_crop_box(args.ppt_crop)
    if not all(math.isfinite(v) and v >= 0 for v in [args.student_offset_sec, args.ppt_offset_sec]):
        raise ValueError('The pretrained timeline requires finite, nonnegative video offsets')
    if args.ppt_source and not args.enable_ocr:
        raise ValueError('Dual PPT video requires --enable-ocr; alternatively use --source and --context-json.')
    pipeline = PretrainedClassroomPipeline(args.config, args.det_weights, args.pose_weights, args.device,
                                          args.context_json, args.task, roi, args.pose_only)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out_dir = args.output.resolve()/f'pretrained_{run_tag}'
    out_dir.mkdir(parents=True, exist_ok=False)
    json_path = (out_dir/Path(args.json_name).name).with_suffix('.json')
    snapshots = SnapshotWriter(json_path)
    frames_dir = out_dir/'frames'
    if args.save_frames:
        frames_dir.mkdir()
    anchor_events = []
    if args.ppt_source:
        anchor_events = extract_anchor_events(args.ppt_source, args.config, args.ppt_interval_sec,
                                              args.ppt_max_frames, args.ppt_offset_sec, crop)
    anchor_index = 0
    cap = open_capture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.
    pipeline.set_fps(fps)
    writer = None
    count = detections = unknown = 0
    next_export_sec = args.student_offset_sec
    last_snapshot = last_frame = None
    last_exported_frame = None
    started = time.perf_counter()

    def export_snapshot(snapshot, frame):
        if args.save_frames:
            frame_path = frames_dir/f'frame_{snapshot.frame_id:07d}.jpg'
            ok, encoded = cv2.imencode('.jpg', frame)
            if not ok:
                raise RuntimeError('Could not encode input frame')
            encoded.tofile(str(frame_path))
            snapshot.frame_image_path = f'frames/{frame_path.name}'
        snapshots.append(snapshot.model_dump(mode='json'))

    try:
        for frame_id, relative_sec, frame in sampled_frames(cap, fps, args.interval_sec, args.max_frames):
            timestamp = relative_sec + args.student_offset_sec
            while anchor_index < len(anchor_events) and anchor_events[anchor_index]['timestamp_sec'] <= timestamp:
                event = anchor_events[anchor_index]
                pipeline.register_anchor(event['entity'], event['timestamp_sec'])
                anchor_index += 1
            snapshot = pipeline.process_frame(frame, frame_id, timestamp,
                                              enable_ocr=args.enable_ocr and not args.ppt_source, ppt_crop=crop)
            snapshot.timestamp = video_timestamp(timestamp)
            last_snapshot, last_frame = snapshot, frame
            if timestamp+1e-9 >= next_export_sec:
                export_snapshot(snapshot, frame)
                last_exported_frame = frame_id
                interval = args.snapshot_interval_sec
                next_export_sec = (args.student_offset_sec + (math.floor(relative_sec/interval)+1)*interval) if interval > 0 else timestamp
            count += 1
            detections += len(snapshot.student_states)
            unknown += sum(s.cas is None for s in snapshot.student_states)
            if (args.save and not args.json_only) or args.show:
                annotated = frame.copy()
                for student in snapshot.student_states:
                    x1,y1,x2,y2 = map(int, student.bbox)
                    color = (180,180,180) if student.cas is None else (180,105,35)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                    value = '--' if student.cas is None else f'{student.cas:.2f}'
                    cv2.putText(annotated, f'T{student.track_id} {student.action.label} rule={value}',
                                (x1,max(16,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                if args.save and not args.json_only:
                    if writer is None:
                        step = max(1, round(fps*args.interval_sec))
                        writer = cv2.VideoWriter(str(out_dir/'observations.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                 fps/step, (frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Could not create output video')
                    writer.write(annotated)
                if args.show:
                    cv2.imshow('Pretrained observations', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            if count == 1 or count % 10 == 0:
                print(f'frame={frame_id} time={timestamp:.2f}s tracks={len(snapshot.student_states)} '
                      f'coverage={snapshot.classroom_metrics.coverage:.1%}', flush=True)
        if not count:
            raise RuntimeError('Input produced no decodable frames; no successful run was published')
        if last_snapshot.frame_id != last_exported_frame:
            export_snapshot(last_snapshot, last_frame)
        snapshots.finish()
    finally:
        snapshots.abort()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    manifest = {
        'completed': True, 'schema_version': '2.0', 'data_source': 'pretrained_inference',
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_name': Path(source).name,
        'analyzed_frames': count, 'snapshots': snapshots.total, 'snapshot_files': snapshots.files,
        'person_observations': detections, 'unknown_score_observations': unknown,
        'elapsed_sec': round(time.perf_counter()-started, 3),
        'input_fps': fps, 'analysis_interval_sec': args.interval_sec,
        'snapshot_interval_sec': args.snapshot_interval_sec,
        'snapshot_segment_max_bytes': snapshots.max_bytes,
        'config': yaml.safe_load(Path(args.config).read_text(encoding='utf-8')),
        'models': pipeline.provenance, 'student_roi': pipeline.roi,
        'context_segments': pipeline.context.segments,
        'validation': 'inference smoke only; no labeled classroom accuracy evaluation',
    }
    (out_dir/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Analyzed {count} frames; saved {snapshots.total} snapshots in {len(snapshots.files)} segment(s): {json_path}', flush=True)
    return out_dir
