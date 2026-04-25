"""Video inference for classroom evaluation.

Modes
-----
Single-stream mode:
    python scripts/infer_video.py --source classroom.mp4 --save

Dual-stream mode:
    python scripts/infer_video.py \
        --student-source students.mp4 \
        --ppt-source ppt.mp4 \
        --save \
        --save-frames

The dual-stream mode aligns student engagement snapshots with PPT OCR anchors
using timestamps on a shared timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_anchor import OCRAnchorDetector
from src.pipeline import ClassroomPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classroom evaluation in single- or dual-video mode.")

    parser.add_argument("--source", type=str, default=None, help="Single classroom video path or camera index.")
    parser.add_argument("--student-source", type=str, default=None, help="Student-facing video path.")
    parser.add_argument("--ppt-source", type=str, default=None, help="PPT/screen-facing video path.")

    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--det-weights", type=str, default=None, help="Detection model weights path (overrides config).")
    parser.add_argument("--pose-weights", type=str, default=None, help="Pose model weights path (overrides config).")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. '0', overrides config).")

    parser.add_argument("--save", action="store_true", help="Save annotated video and JSON output.")
    parser.add_argument("--save-frames", action="store_true", help="Save sampled student frames for the frontend image player.")
    parser.add_argument("--frames-dir-name", type=str, default="frames", help="Frame image subdirectory name under --output.")
    parser.add_argument("--json-name", type=str, default="snapshots.json", help="Stable JSON filename written inside --output.")
    parser.add_argument("--anchor-json-name", type=str, default="anchor_events.json", help="Anchor event JSON filename for dual-stream mode.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/latest"))

    parser.add_argument("--show", action="store_true", help="Display live preview window.")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="Analyze one student frame every N seconds.")
    parser.add_argument("--ppt-interval-sec", type=float, default=1.0, help="Run OCR on one PPT frame every N seconds in dual-stream mode.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most N sampled student frames (0=all).")
    parser.add_argument("--ppt-max-frames", type=int, default=0, help="Process at most N sampled PPT frames (0=all).")

    parser.add_argument("--student-offset-sec", type=float, default=0.0, help="Seconds added to the student video timeline before alignment.")
    parser.add_argument("--ppt-offset-sec", type=float, default=0.0, help="Seconds added to the PPT video timeline before alignment.")
    parser.add_argument(
        "--ppt-crop",
        type=str,
        default=None,
        help="Optional PPT crop box as x1,y1,x2,y2 for OCR.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    single_mode = bool(args.source)
    dual_mode = bool(args.student_source and args.ppt_source)

    if single_mode and dual_mode:
        raise ValueError("Use either --source or (--student-source and --ppt-source), not both.")

    if not single_mode and not dual_mode:
        raise ValueError("You must provide --source or both --student-source and --ppt-source.")


def video_timestamp(seconds: float) -> str:
    base = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def cas_color(cas: float) -> tuple[int, int, int]:
    if cas >= 0.8:
        return (0, 200, 0)
    if cas >= 0.6:
        return (0, 200, 200)
    return (0, 0, 200)


def open_capture(source: str):
    resolved_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(resolved_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def parse_crop_box(raw_crop: str | None) -> tuple[int, int, int, int] | None:
    if not raw_crop:
        return None

    parts = [part.strip() for part in raw_crop.split(",")]
    if len(parts) != 4:
        raise ValueError("--ppt-crop must be in x1,y1,x2,y2 format.")

    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--ppt-crop must contain integers only.") from exc

    return x1, y1, x2, y2


def clip_crop(frame, crop_box: tuple[int, int, int, int] | None):
    if crop_box is None:
        return frame

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = crop_box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return frame

    return frame[y1:y2, x1:x2]


def annotate_frame(frame, snapshot) -> tuple:
    annotated = frame.copy()

    for student in snapshot.student_states:
        x1, y1, x2, y2 = [int(value) for value in student.bbox]
        color = cas_color(student.cas)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"T{student.track_id} {student.action.label} CAS={student.cas:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    ctes = snapshot.classroom_metrics.ctes_score
    cv2.putText(
        annotated,
        f"CTES={ctes:.3f}  N={snapshot.classroom_metrics.active_tracks}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return annotated, ctes


def sampled_frames(
    cap,
    fps: float,
    interval_sec: float,
    max_frames: int,
) -> Iterable[tuple[int, float, any]]:
    frame_idx = 0
    processed_count = 0
    frame_step = max(1, int(round(fps * interval_sec)))

    while True:
        frame_idx += 1
        should_analyze = interval_sec <= 0 or (frame_idx - 1) % frame_step == 0

        if not should_analyze:
            # Opt: grab() moves the cursor without full decode (~60% faster)
            if not cap.grab():
                break
            continue

        ret, frame = cap.read()
        if not ret:
            break

        processed_count += 1
        if 0 < max_frames < processed_count:
            break

        timestamp_sec = (frame_idx - 1) / fps
        yield frame_idx, timestamp_sec, frame


def _flush_snapshots(snapshots: list[dict], json_path: Path) -> None:
    """Append snapshots to a JSON Lines file incrementally."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "a", encoding="utf-8") as f:
        for snap in snapshots:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")


def load_vsam_config(config_path: Path) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload.get("vsam", {})


def extract_anchor_events(
    ppt_source: str,
    config_path: Path,
    interval_sec: float,
    max_frames: int,
    timeline_offset_sec: float,
    crop_box: tuple[int, int, int, int] | None,
):
    vsam_config = load_vsam_config(config_path)
    detector = OCRAnchorDetector(
        change_threshold=vsam_config.get("text_change_threshold", 0.3),
    )

    cap = open_capture(ppt_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    events: list[dict] = []

    print(f"Scanning PPT stream: {ppt_source}  (fps={fps:.2f})")

    try:
        for frame_idx, relative_sec, frame in sampled_frames(cap, fps, interval_sec, max_frames):
            merged_sec = timeline_offset_sec + relative_sec
            crop = clip_crop(frame, crop_box)
            detected_text = detector.detect_change(crop)
            if detected_text is None:
                continue

            entity = detected_text[:60].strip()
            if not entity:
                continue

            events.append({
                "entity": entity,
                "timestamp_sec": round(merged_sec, 4),
                "ppt_frame_id": frame_idx,
                "timestamp": video_timestamp(merged_sec),
            })
    finally:
        cap.release()

    print(f"Detected {len(events)} PPT knowledge anchors")
    return events


def run_single_stream(args: argparse.Namespace) -> None:
    pipeline = ClassroomPipeline(
        config_path=args.config,
        det_weights=args.det_weights,
        pose_weights=args.pose_weights,
        device=args.device,
    )

    cap = open_capture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline.set_fps(fps)

    writer = None
    out_video: Path | None = None
    frames_dir: Path | None = None

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_stem = Path(args.source).stem if not args.source.isdigit() else "cam"
    out_name = f"{source_stem}_{run_tag}"

    if args.save:
        args.output.mkdir(parents=True, exist_ok=True)
        out_video = args.output / f"{out_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Output video uses native fps for smooth playback
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

        if args.save_frames:
            frames_dir = args.output / args.frames_dir_name
            frames_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict] = []
    last_ctes = 0.0

    # Clear stale output from previous runs
    if args.save:
        json_path = args.output / args.json_name
        if json_path.exists():
            json_path.unlink()

    print(f"Processing single stream: {args.source}  ({width}x{height} @ {fps:.1f} fps)")

    # Determine analysis interval in frames (1 analysis per interval_sec)
    analysis_step = max(1, int(round(fps * args.interval_sec))) if args.interval_sec > 0 else 1
    frame_idx = 0
    analyzed_count = 0
    latest_snapshot = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            relative_sec = (frame_idx - 1) / fps
            merged_sec = args.student_offset_sec + relative_sec

            # Decide whether this frame triggers a full pipeline analysis
            should_analyze = (frame_idx - 1) % analysis_step == 0

            if should_analyze:
                analyzed_count += 1
                if 0 < args.max_frames < analyzed_count:
                    break

                snapshot = pipeline.process_frame(
                    frame, frame_id=frame_idx,
                    timestamp_sec=merged_sec, enable_ocr=True,
                )
                snapshot.timestamp = video_timestamp(merged_sec)
                latest_snapshot = snapshot

                if frames_dir is not None:
                    frame_name = f"frame_{frame_idx:06d}.jpg"
                    frame_file = frames_dir / frame_name
                    cv2.imwrite(str(frame_file), frame)
                    snapshot.frame_image_path = (
                        f"{args.frames_dir_name}/{frame_name}".replace("\\", "/")
                    )

                snapshots.append(snapshot.model_dump())

                if frame_idx % 100 == 0:
                    print(f"  frame {frame_idx}  CTES={last_ctes:.3f}")

                # Flush to disk periodically to avoid OOM on long videos
                if args.save and len(snapshots) >= 500:
                    _flush_snapshots(snapshots, args.output / args.json_name)
                    snapshots.clear()

            # ── Always write annotated frame to the output video ──
            if latest_snapshot is not None:
                annotated, last_ctes = annotate_frame(frame, latest_snapshot)
            else:
                annotated = frame  # before first analysis, write raw frame

            if writer:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Classroom Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if args.save:
        json_path = args.output / args.json_name
        # Flush remaining snapshots
        if snapshots:
            _flush_snapshots(snapshots, json_path)
        print(f"Saved snapshots to {json_path}")
        if out_video is not None:
            print(f"Annotated video saved to {out_video}")
        if frames_dir is not None:
            print(f"Frame images saved to {frames_dir}")


def run_dual_stream(args: argparse.Namespace) -> None:
    ppt_crop = parse_crop_box(args.ppt_crop)
    anchor_events = extract_anchor_events(
        ppt_source=args.ppt_source,
        config_path=args.config,
        interval_sec=args.ppt_interval_sec,
        max_frames=args.ppt_max_frames,
        timeline_offset_sec=args.ppt_offset_sec,
        crop_box=ppt_crop,
    )

    pipeline = ClassroomPipeline(
        config_path=args.config,
        det_weights=args.det_weights,
        pose_weights=args.pose_weights,
        device=args.device,
    )

    cap = open_capture(args.student_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline.set_fps(fps)

    writer = None
    out_video: Path | None = None
    frames_dir: Path | None = None

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    student_stem = Path(args.student_source).stem
    ppt_stem = Path(args.ppt_source).stem
    out_name = f"{student_stem}__{ppt_stem}_{run_tag}"

    if args.save:
        args.output.mkdir(parents=True, exist_ok=True)
        out_video = args.output / f"{out_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Output video uses native fps for smooth playback
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

        if args.save_frames:
            frames_dir = args.output / args.frames_dir_name
            frames_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict] = []
    last_ctes = 0.0
    anchor_index = 0

    print(
        "Processing dual stream: "
        f"students={args.student_source}, ppt={args.ppt_source}  "
        f"({width}x{height} @ {fps:.1f} fps)"
    )

    # Determine analysis interval in frames (1 analysis per interval_sec)
    analysis_step = max(1, int(round(fps * args.interval_sec))) if args.interval_sec > 0 else 1
    frame_idx = 0
    analyzed_count = 0
    latest_snapshot = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            relative_sec = (frame_idx - 1) / fps
            merged_sec = args.student_offset_sec + relative_sec

            # Inject any pending anchor events up to current timestamp
            while anchor_index < len(anchor_events) and anchor_events[anchor_index]["timestamp_sec"] <= merged_sec:
                event = anchor_events[anchor_index]
                pipeline.register_anchor(event["entity"], event["timestamp_sec"])
                anchor_index += 1

            # Decide whether this frame triggers a full pipeline analysis
            should_analyze = (frame_idx - 1) % analysis_step == 0

            if should_analyze:
                analyzed_count += 1
                if 0 < args.max_frames < analyzed_count:
                    break

                snapshot = pipeline.process_student_frame(
                    frame=frame,
                    frame_id=frame_idx,
                    timestamp_sec=merged_sec,
                )
                snapshot.timestamp = video_timestamp(merged_sec)
                latest_snapshot = snapshot

                if frames_dir is not None:
                    frame_name = f"frame_{frame_idx:06d}.jpg"
                    frame_file = frames_dir / frame_name
                    cv2.imwrite(str(frame_file), frame)
                    snapshot.frame_image_path = (
                        f"{args.frames_dir_name}/{frame_name}".replace("\\", "/")
                    )

                snapshots.append(snapshot.model_dump())

                if frame_idx % 100 == 0:
                    print(f"  student frame {frame_idx}  CTES={last_ctes:.3f}")

            # ── Always write annotated frame to the output video ──
            if latest_snapshot is not None:
                annotated, last_ctes = annotate_frame(frame, latest_snapshot)
            else:
                annotated = frame  # before first analysis, write raw frame

            if writer:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Classroom Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if args.save:
        json_path = args.output / args.json_name
        json_path.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(snapshots)} fused snapshots to {json_path}")

        anchor_path = args.output / args.anchor_json_name
        anchor_path.write_text(json.dumps(anchor_events, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(anchor_events)} anchor events to {anchor_path}")

        if out_video is not None:
            print(f"Annotated student video saved to {out_video}")
        if frames_dir is not None:
            print(f"Frame images saved to {frames_dir}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.source:
        run_single_stream(args)
        return

    run_dual_stream(args)


if __name__ == "__main__":
    main()
