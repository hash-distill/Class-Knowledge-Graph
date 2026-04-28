"""Generate demo classroom snapshots for the frontend dashboard.

This script does not run CV inference. It creates a structured snapshots.json
using knowledge points from the existing graph and synthetic student states so
the dashboard, graph sync and AI module can be wired end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_GRAPH_PATH = Path("../knowledge_graph/china_primary_school_math_knowledge_graph.json")
DEFAULT_OUTPUT = Path("artifacts/results/demo/snapshots.json")
DEFAULT_IMAGE = "frames/demo.png"
DEFAULT_GRADE = "小学五年级上册"

ACTION_POOL = [
    ("hand_raising", 0.95),
    ("writing", 0.80),
    ("attending", 0.75),
    ("attending", 0.70),
    ("looking_around", 0.25),
    ("leaning", 0.15),
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


def build_student_state(track_id: int, frame_index: int, rng: random.Random, mean_bias: float) -> dict:
    action_label, action_score = ACTION_POOL[(track_id + frame_index) % len(ACTION_POOL)]
    gaze_score = max(0.18, min(0.95, mean_bias + rng.uniform(-0.12, 0.12)))
    cas = max(0.1, min(0.98, action_score * 0.68 + gaze_score * 0.32 + rng.uniform(-0.05, 0.05)))

    column = (track_id - 1) % 6
    row = (track_id - 1) // 6
    left = 110 + column * 255 + rng.uniform(-16, 16)
    top = 320 + row * 165 + rng.uniform(-10, 10)
    width = 138 + rng.uniform(-8, 12)
    height = 248 + rng.uniform(-12, 12)

    focus_zone = "board_focus" if cas >= 0.75 else ("desk_focus" if cas >= 0.55 else "wandering")

    return {
        "track_id": track_id,
        "bbox": [
            round(left, 2),
            round(top, 2),
            round(left + width, 2),
            round(top + height, 2),
        ],
        "action": {
            "label": action_label,
            "confidence": round(max(0.2, min(0.99, action_score + rng.uniform(-0.08, 0.06))), 4),
            "engagement_score": round(action_score, 4),
            "source": "demo",
        },
        "gaze": {
            "pitch": round(rng.uniform(-18, 12), 2),
            "yaw": round(rng.uniform(-16, 16), 2),
            "roll": round(rng.uniform(-8, 8), 2),
            "focus_score": round(gaze_score, 4),
            "focus_zone": focus_zone,
            "source": "demo",
        },
        "cas": round(cas, 4),
    }


def build_snapshot(
    frame_index: int,
    timestamp: datetime,
    knowledge_node: dict,
    students: list[dict],
    image_path: str,
) -> dict:
    cas_values = [student["cas"] for student in students]
    mean_cas = sum(cas_values) / len(cas_values)
    variance = sum((value - mean_cas) ** 2 for value in cas_values) / len(cas_values)
    std_cas = math.sqrt(variance)
    ctes = mean_cas * math.exp(-0.65 * std_cas)

    behavior_counts = Counter(student["action"]["label"] for student in students)

    return {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "frame_id": frame_index,
        "frame_image_path": image_path,
        "knowledge_anchor": {
            "entity": knowledge_node["node_name"],
            "uuid": knowledge_node["uuid"],
            "grade": knowledge_node["grade"],
            "score_k": round(mean_cas, 4),
            "visual_score": round(mean_cas, 4),
            "trigger_time": f"{(frame_index - 1) * 2:.1f}s",
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


def generate_demo_snapshots(
    graph_path: Path,
    output_path: Path,
    image_path: str,
    grade: str,
    frames: int,
    students_per_frame: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    knowledge_nodes = load_knowledge_nodes(graph_path, grade)
    chosen_nodes = rng.sample(knowledge_nodes, k=min(6, len(knowledge_nodes)))
    if not chosen_nodes:
        raise ValueError("Unable to choose demo knowledge nodes")

    timeline_start = datetime(2026, 4, 24, 8, 0, 0, tzinfo=timezone.utc)
    snapshots: list[dict] = []

    segment_length = max(1, frames // len(chosen_nodes))
    for frame_index in range(1, frames + 1):
        knowledge_node = chosen_nodes[min((frame_index - 1) // segment_length, len(chosen_nodes) - 1)]
        mean_bias = 0.58 + 0.14 * math.sin(frame_index / 18)
        students = [
            build_student_state(track_id=index + 1, frame_index=frame_index, rng=rng, mean_bias=mean_bias)
            for index in range(students_per_frame)
        ]
        timestamp = timeline_start + timedelta(seconds=(frame_index - 1) * 2)
        snapshots.append(
            build_snapshot(
                frame_index=frame_index,
                timestamp=timestamp,
                knowledge_node=knowledge_node,
                students=students,
                image_path=image_path,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo snapshots for the dashboard.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--grade", type=str, default=DEFAULT_GRADE)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--students", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260424)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = generate_demo_snapshots(
        graph_path=args.graph,
        output_path=args.output,
        image_path=args.image_path,
        grade=args.grade,
        frames=args.frames,
        students_per_frame=args.students,
        seed=args.seed,
    )
    print(f"Generated {len(snapshots)} demo snapshots -> {args.output}")
    if snapshots:
        first_anchor = snapshots[0]["knowledge_anchor"]["entity"]
        print(f"First knowledge anchor: {first_anchor}")


if __name__ == "__main__":
    main()
