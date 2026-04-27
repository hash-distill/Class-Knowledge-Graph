"""Smoke test: verify the full pipeline with mock data (no GPU required).

Usage::

    python scripts/smoke_test.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import (
    ActionRecord,
    ActionSource,
    BBoxRecord,
    ClassroomSnapshot,
    GazeRecord,
    GazeSource,
    FocusZone,
    KnowledgeAnchor,
    StudentState,
)
from src.scoring import calc_cas, calc_ctes, compute_classroom_metrics
from src.vsam import gaussian_weight, score_knowledge_point, VSAMAligner


def mock_students(n: int = 5) -> list[StudentState]:
    """Create synthetic student states for testing."""
    states: list[StudentState] = []
    for i in range(n):
        action_conf = 0.5 + 0.1 * i
        gaze_score = 0.4 + 0.12 * i
        cas = calc_cas(action_conf, gaze_score)
        states.append(
            StudentState(
                track_id=i + 1,
                bbox=[100.0 + i * 120, 100.0, 200.0 + i * 120, 350.0],
                action=ActionRecord(
                    label=["focus_student", "focus_student", "active_student", "distracted_student", "active_student"][i % 5],
                    confidence=round(action_conf, 4),
                    source=ActionSource.DETECTION,
                ),
                gaze=GazeRecord(
                    pitch=-10.0 + i * 5,
                    yaw=5.0 + i * 3,
                    focus_score=round(gaze_score, 4),
                    focus_zone=[FocusZone.BOARD, FocusZone.DESK, FocusZone.BOARD,
                                FocusZone.WANDERING, FocusZone.BOARD][i % 5],
                    source=GazeSource.PNP,
                ),
                cas=round(cas, 4),
            )
        )
    return states


def run_smoke(output_json: Path) -> dict:
    """Run full mock pipeline and produce a snapshot."""
    print("=" * 60)
    print("  SMOKE TEST: Mock Pipeline Verification")
    print("=" * 60)

    # 1. Mock student states
    students = mock_students(5)
    print(f"\n[1] Created {len(students)} mock student states")
    for s in students:
        print(f"    Track {s.track_id}: action={s.action.label}({s.action.confidence:.2f})"
              f"  gaze={s.gaze.focus_score:.2f}({s.gaze.focus_zone.value})"
              f"  CAS={s.cas:.2f}")

    # 2. Classroom metrics
    metrics = compute_classroom_metrics(students, lambda_penalty=1.0)
    print(f"\n[2] Classroom Metrics:")
    print(f"    CTES  = {metrics.ctes_score:.4f}")
    print(f"    μ_CAS = {metrics.mean_cas:.4f}")
    print(f"    σ_CAS = {metrics.std_cas:.4f}")
    print(f"    Behaviors: {metrics.behavior_distribution.counts}")

    # 3. VSAM alignment test
    print(f"\n[3] VSAM Gaussian Alignment:")
    aligner = VSAMAligner(mu=3.0, sigma=1.5, window_duration=10.0)
    aligner.trigger("分数加减法", t_ocr=5.0)

    for t in range(6, 18):
        fake_cas = 0.5 + 0.3 * gaussian_weight(t, 5.0, mu=3.0, sigma=1.5)
        aligner.feed(float(t), fake_cas)

    closed = aligner.evaluate(current_time=18.0)
    anchor = KnowledgeAnchor()
    if closed:
        a = closed[0]
        anchor = KnowledgeAnchor(
            entity=a.entity,
            trigger_time=f"{a.t_ocr:.1f}s",
            gaussian_weight=round(gaussian_weight(8.0, 5.0), 4),
            score_k=round(a.score_k, 4),
        )
        print(f"    Knowledge: {a.entity}")
        print(f"    Score_Ki = {a.score_k:.4f}")
    else:
        print("    (no closed anchors yet)")

    # 4. Build snapshot
    snapshot = ClassroomSnapshot(
        timestamp="2026-04-15T10:15:30Z",
        frame_id=750,
        knowledge_anchor=anchor,
        classroom_metrics=metrics,
        student_states=students,
    )

    payload = snapshot.model_dump()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n[4] Snapshot saved to: {output_json}")
    print(f"    Total students: {len(students)}")
    print(f"    CTES: {metrics.ctes_score:.4f}")
    print("\n" + "=" * 60)
    print("  ✅ SMOKE TEST PASSED")
    print("=" * 60)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for classroom pipeline.")
    p.add_argument("--mock", action="store_true", help="Run with synthetic data.")
    p.add_argument("--output", type=Path, default=Path("artifacts/smoke/snapshot.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mock:
        print("Use --mock flag to run smoke test with synthetic data.")
        return
    run_smoke(args.output)


if __name__ == "__main__":
    main()
