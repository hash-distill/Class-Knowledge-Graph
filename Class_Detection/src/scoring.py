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
    w_action: float = 1.0,
    w_gaze: float = 1.0,
) -> float:
    """Classroom Attention Score per student.

    CAS = max(w1 * S_action, w2 * S_gaze)

    Uses ``max`` instead of weighted average so that two valid but
    asymmetric signals (e.g. high-action + low-gaze when writing)
    do not dilute each other.
    """
    return _clamp01(max(w_action * action_score, w_gaze * gaze_score))


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
