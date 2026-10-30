"""CAS / CTES scoring engine.

CAS  = Classroom Attention Score (per-student)
CTES = Classroom Teaching Effect Score (whole-class)
"""

from __future__ import annotations

import math
from typing import Optional

from src.schema import (
    BehaviorDistribution,
    ClassroomMetrics,
    StudentState,
)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ── Individual CAS ───────────────────────────────────────────

def calc_cas(
    action_score: float,
    gaze_score: float,
    action_label: str = "",
    w_action: float = 0.6,
    w_gaze: float = 0.4,
) -> float:
    """Classroom Attention Score per student.

    CAS = max(w_action · S_action, w_gaze · S_gaze)

    Uses max-fusion instead of weighted average to prevent signal
    dilution: a student actively writing (high S_action) but looking
    down (low S_gaze) should still be considered engaged.

    Applies a severe penalty coefficient if the student is engaged
    in obviously negative behaviors (e.g. using phone, yawning).
    """
    base_cas = max(w_action * float(action_score), w_gaze * float(gaze_score))

    # 负面行为惩罚
    penalty_factor = 1.0
    negative_labels = ["distracted", "using_phone", "yawning", "sleeping",
                       "lean_desk", "use_phone", "yawn"]
    if any(neg in action_label.lower() for neg in negative_labels):
        penalty_factor = 0.5  # 显著降低专注度得分

    return _clamp01(base_cas * penalty_factor)


# ── Class-level CTES ─────────────────────────────────────────

def calc_ctes(
    cas_values: list[float],
    lambda_penalty: float = 1.0,
) -> float:
    """Classroom Teaching Effect Score.

    CTES = μ_CAS · exp(−λ · σ_CAS)

    Variance penalty exponentially down-weights polarized classrooms
    where a few students are highly engaged and others are completely
    disengaged.
    """
    if not cas_values:
        return 0.0

    vals = [_clamp01(v) for v in cas_values]
    n = len(vals)
    mean_val = sum(vals) / n
    variance = sum((v - mean_val) ** 2 for v in vals) / n
    std_val = math.sqrt(variance)
    return _clamp01(mean_val * math.exp(-lambda_penalty * std_val))


# ── Aggregate classroom snapshot ─────────────────────────────

def compute_classroom_metrics(
    student_states: list[StudentState],
    lambda_penalty: float = 1.0,
) -> ClassroomMetrics:
    """Build ClassroomMetrics from a list of per-student states."""
    if not student_states:
        return ClassroomMetrics()

    cas_vals = [s.cas for s in student_states]
    n = len(cas_vals)
    mean_cas = sum(cas_vals) / n
    std_cas = math.sqrt(sum((v - mean_cas) ** 2 for v in cas_vals) / n)
    ctes = calc_ctes(cas_vals, lambda_penalty)

    # Behaviour distribution
    label_counts: dict[str, int] = {}
    for s in student_states:
        label = s.action.label
        label_counts[label] = label_counts.get(label, 0) + 1

    return ClassroomMetrics(
        ctes_score=round(ctes, 4),
        mean_cas=round(mean_cas, 4),
        std_cas=round(std_cas, 4),
        active_tracks=n,
        behavior_distribution=BehaviorDistribution(counts=label_counts),
    )
