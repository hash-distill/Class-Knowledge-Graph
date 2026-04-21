"""Video inference: run the full classroom pipeline on a video file.

Usage::

    python scripts/infer_video.py \\
        --source classroom.mp4 \\
        --config configs/pipeline.yaml \\
        --save --output artifacts/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ClassroomPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full classroom pipeline on a video.")
    p.add_argument("--source", type=str, required=True, help="Video path or camera index.")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    p.add_argument("--det-weights", type=str, default=None, help="Detection model weights path (overrides config).")
    p.add_argument("--pose-weights", type=str, default=None, help="Pose model weights path (overrides config).")
    p.add_argument("--device", type=str, default=None, help="Device to run on (e.g. '0', overrides config).")
    p.add_argument("--save", action="store_true", help="Save annotated video + JSON.")
    p.add_argument("--output", type=Path, default=Path("artifacts/results"))
    p.add_argument("--show", action="store_true", help="Display live preview window.")
    p.add_argument("--interval-sec", type=float, default=1.0, help="Analyze 1 frame every N seconds to save compute.")
    p.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0=all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ClassroomPipeline(
        config_path=args.config,
        det_weights=args.det_weights,
        pose_weights=args.pose_weights,
        device=args.device,
    )

    # Open video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline.set_fps(fps)

    writer = None
    # Use timestamp to create unique output filenames, preventing overwrites
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_stem = Path(args.source).stem if not args.source.isdigit() else "cam"
    out_name = f"{source_stem}_{run_tag}"

    # Output video should always run at the source original FPS
    output_fps = fps

    if args.save:
        args.output.mkdir(parents=True, exist_ok=True)
        out_video = args.output / f"{out_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, output_fps, (w, h))

    snapshots: list[dict] = []
    frame_idx = 0
    processed_count = 0

    print(f"Processing: {args.source}  ({w}x{h} @ {fps:.1f} fps)")

    frame_step = max(1, int(fps * args.interval_sec))

    while True:
        # Opt-1: Use grab() to skip decoding for non-analyzed frames
        if not cap.grab():
            break
        
        frame_idx += 1
        
        # Decide if we analyze this frame
        if args.interval_sec > 0 and (frame_idx - 1) % frame_step != 0:
            continue
            
        # Bug-3: Use processed_count for max_frames limit
        processed_count += 1
        if 0 < args.max_frames < processed_count:
            break

        # Actually decode the frame we want to process
        ret, frame = cap.retrieve()
        if not ret:
            break

        # Bug-5: Pass frame_idx to pipeline to ensure stable timing
        snapshot = pipeline.process_frame(frame, frame_id=frame_idx)
        
        # Only record JSON representation once per second
        if frame_idx % max(1, int(fps)) == 0:
            snapshots.append(snapshot.model_dump())

        # Draw basic overlay
        annotated = frame.copy()
        for s in snapshot.student_states:
            x1, y1, x2, y2 = [int(v) for v in s.bbox]
            color = _cas_color(s.cas)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"T{s.track_id} {s.action.label} CAS={s.cas:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # CTES overlay
        ctes = snapshot.classroom_metrics.ctes_score
        cv2.putText(annotated, f"CTES={ctes:.3f}  N={snapshot.classroom_metrics.active_tracks}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Classroom Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx} (processed {processed_count})  CTES={ctes:.3f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if args.save:
        json_path = args.output / f"{out_name}.json"
        json_path.write_text(
            json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(snapshots)} snapshots to {json_path}")
        print(f"Annotated video saved to {out_video}")

    print(f"Done. Processed {processed_count} frames (video index {frame_idx}).")


def _cas_color(cas: float) -> tuple[int, int, int]:
    """Map CAS [0,1] to BGR color: red(low) → yellow(mid) → green(high)."""
    if cas > 0.7:
        return (0, 200, 0)
    elif cas > 0.4:
        return (0, 200, 200)
    else:
        return (0, 0, 200)


if __name__ == "__main__":
    main()
