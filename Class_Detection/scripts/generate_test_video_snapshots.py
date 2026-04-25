"""Generate dashboard snapshots from test.mp4 using dense frame extraction.

This script extracts image frames with ffmpeg for smooth image-sequence playback
on the dashboard right panel. Knowledge points are sampled from the graph and
student engagement is deterministically computed from track_id and frame index.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))

import imageio_ffmpeg


DEFAULT_INPUT = PROJECT_ROOT / "test.mp4"
DEFAULT_GRAPH_PATH = Path(r"D:\4C\ai生成和知识后端\china_primary_school_math_knowledge_graph.json")
DEFAULT_OUTPUT = Path(r"D:\4C\前端\Front\public\data\visual\latest\snapshots.json")
DEFAULT_FRAMES_DIR = Path(r"D:\4C\前端\Front\public\data\visual\latest\frames")
DEFAULT_GRADE = "小学五年级上册"

ACTION_POOL = [
    ("hand_raising", 0.93),
    ("read", 0.72),
    ("write", 0.78),
    ("discuss", 0.84),
    ("attend", 0.88),
    ("wander", 0.36),
]


def load_knowledge_nodes(graph_path: Path, grade: str) -> list[dict]:
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    filtered = [
        node["properties"]
        for node in nodes
        if (node.get("properties") or {}).get("grade") == grade
    ]
    if not filtered:
        raise ValueError(f"No graph nodes found for grade={grade}")
    return filtered


def load_video_meta(video_path: Path) -> dict:
    frame_count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    frame_reader = imageio_ffmpeg.read_frames(str(video_path))
    meta = next(frame_reader)
    frame_reader.close()

    meta["frame_count"] = int(frame_count)
    meta["duration"] = float(seconds)
    return meta


def clear_old_frames(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("test_*.jpg"):
        path.unlink(missing_ok=True)


def extract_frames(video_path: Path, frames_dir: Path, sample_fps: float) -> list[Path]:
    clear_old_frames(frames_dir)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_pattern = str(frames_dir / "test_%06d.jpg")
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps}",
        "-q:v",
        "6",
        output_pattern,
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(frames_dir.glob("test_*.jpg"))
    if not frames:
        raise ValueError(f"No extracted frames generated from {video_path}")
    return frames


def build_student_state(
    track_id: int,
    frame_index: int,
    width: int,
    height: int,
    rng: random.Random,
) -> dict:
    action_label, action_score = ACTION_POOL[(track_id + frame_index) % len(ACTION_POOL)]
    temporal_wave = 0.14 * math.sin(frame_index / 18 + track_id / 4)
    focus_score = max(0.18, min(0.96, 0.56 + temporal_wave + rng.uniform(-0.05, 0.05)))
    cas = max(0.1, min(0.99, action_score * 0.68 + focus_score * 0.32 + rng.uniform(-0.02, 0.02)))

    cols = 6
    row = (track_id - 1) // cols
    col = (track_id - 1) % cols
    base_left = width * (0.08 + col * 0.145)
    base_top = height * (0.58 + row * 0.14)
    box_width = width * 0.09
    box_height = height * 0.26

    jitter_x = math.sin(frame_index / 13 + track_id) * width * 0.004
    jitter_y = math.cos(frame_index / 9 + track_id) * height * 0.003
    left = max(0.0, base_left + jitter_x)
    top = max(0.0, base_top + jitter_y)

    focus_zone = "board_focus" if cas >= 0.75 else ("desk_focus" if cas >= 0.55 else "wandering")

    return {
        "track_id": track_id,
        "bbox": [
            round(left, 2),
            round(top, 2),
            round(min(width - 1, left + box_width), 2),
            round(min(height - 1, top + box_height), 2),
        ],
        "action": {
            "label": action_label,
            "confidence": round(max(0.2, min(0.99, action_score + rng.uniform(-0.05, 0.05))), 4),
            "engagement_score": round(action_score, 4),
            "source": "video-demo",
        },
        "gaze": {
            "pitch": round(rng.uniform(-16, 12), 2),
            "yaw": round(rng.uniform(-14, 14), 2),
            "roll": round(rng.uniform(-7, 7), 2),
            "focus_score": round(focus_score, 4),
            "focus_zone": focus_zone,
            "source": "video-demo",
        },
        "cas": round(cas, 4),
    }


def build_snapshot(
    frame_path: str,
    frame_id: int,
    timestamp_sec: float,
    knowledge_node: dict,
    students: list[dict],
) -> dict:
    cas_values = [student["cas"] for student in students]
    mean_cas = sum(cas_values) / len(cas_values)
    variance = sum((value - mean_cas) ** 2 for value in cas_values) / len(cas_values)
    std_cas = math.sqrt(variance)
    ctes = mean_cas * math.exp(-0.65 * std_cas)
    behavior_counts = Counter(student["action"]["label"] for student in students)

    base_time = datetime(2026, 4, 24, 8, 0, 0, tzinfo=timezone.utc)
    timestamp = (base_time + timedelta(seconds=timestamp_sec)).isoformat().replace("+00:00", "Z")

    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "frame_image_path": frame_path,
        "knowledge_anchor": {
            "entity": knowledge_node["node_name"],
            "uuid": knowledge_node["uuid"],
            "grade": knowledge_node["grade"],
            "score_k": round(mean_cas, 4),
            "visual_score": round(mean_cas, 4),
            "trigger_time": f"{timestamp_sec:.1f}s",
            "gaussian_weight": round(max(cas_values) - min(cas_values), 4),
        },
        "classroom_metrics": {
            "ctes_score": round(ctes, 4),
            "mean_cas": round(mean_cas, 4),
            "std_cas": round(std_cas, 4),
            "active_tracks": len(students),
            "behavior_distribution": dict(behavior_counts),
        },
        "student_states": students,
    }


def generate_snapshots(
    video_path: Path,
    graph_path: Path,
    output_path: Path,
    frames_dir: Path,
    grade: str,
    students_per_frame: int,
    sample_fps: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    meta = load_video_meta(video_path)
    width, height = meta["size"]
    source_fps = float(meta["fps"])
    frame_paths = extract_frames(video_path, frames_dir, sample_fps)

    knowledge_nodes = load_knowledge_nodes(graph_path, grade)
    chosen_nodes = rng.sample(knowledge_nodes, k=min(6, len(knowledge_nodes)))
    segment_length = max(1, len(frame_paths) // len(chosen_nodes))

    snapshots: list[dict] = []
    for index, frame_file in enumerate(frame_paths, start=1):
        knowledge_node = chosen_nodes[min((index - 1) // segment_length, len(chosen_nodes) - 1)]
        timestamp_sec = (index - 1) / sample_fps
        frame_id = max(1, int(round(timestamp_sec * source_fps)) + 1)

        students = [
            build_student_state(
                track_id=student_index + 1,
                frame_index=index,
                width=width,
                height=height,
                rng=rng,
            )
            for student_index in range(students_per_frame)
        ]

        snapshots.append(
            build_snapshot(
                frame_path=f"frames/{frame_file.name}",
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                knowledge_node=knowledge_node,
                students=students,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate snapshots from a real mp4 for the dashboard.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--grade", type=str, default=DEFAULT_GRADE)
    parser.add_argument("--students", type=int, default=18)
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260424)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = generate_snapshots(
        video_path=args.input,
        graph_path=args.graph,
        output_path=args.output,
        frames_dir=args.frames_dir,
        grade=args.grade,
        students_per_frame=args.students,
        sample_fps=args.sample_fps,
        seed=args.seed,
    )
    print(f"Generated {len(snapshots)} real-frame snapshots -> {args.output}")
    if snapshots:
        print(
            f"First snapshot frame_id={snapshots[0]['frame_id']}, "
            f"knowledge={snapshots[0]['knowledge_anchor']['entity']}"
        )


if __name__ == "__main__":
    main()
