"""Auditable, time-based rules over observations; no learned engagement claims."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.schema import BehaviorDistribution, ClassroomMetrics, ObservationAssessment


def canonical_behavior(name: str) -> str:
    key = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    return {
        'read': 'read', 'reading': 'read', 'write': 'write', 'writing': 'write',
        'raise_hand': 'raise_hand', 'hand_raising': 'raise_hand',
        'hand_raise': 'raise_hand', 'raising_hand': 'raise_hand',
        'raising_hands': 'raise_hand', 'handraising': 'raise_hand',
    }.get(key, 'unknown')


def match_behaviors(people, behaviors, min_iou=.2) -> dict[int, int]:
    """One-to-one association. Permit an upper-body box contained in a person."""
    if not len(people) or not len(behaviors):
        return {}
    affinity = np.zeros((len(people), len(behaviors)))
    valid = np.zeros_like(affinity, dtype=bool)
    for i, p in enumerate(people):
        pa = max(0., p[2]-p[0]) * max(0., p[3]-p[1])
        for j, b in enumerate(behaviors):
            ba = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
            intersection = max(0., min(p[2], b[2])-max(p[0], b[0])) * max(0., min(p[3], b[3])-max(p[1], b[1]))
            iou = intersection / max(pa+ba-intersection, 1e-8)
            containment = intersection / max(min(pa, ba), 1e-8)
            close = abs((p[0]+p[2]-b[0]-b[2])/2) < .35 * max(p[2]-p[0], 1)
            valid[i, j] = iou >= min_iou or (containment >= .7 and close and ba <= pa)
            affinity[i, j] = .6 * iou + .4 * containment
    rows, cols = linear_sum_assignment(np.where(valid, -affinity, 1e6))
    return {int(i): int(j) for i, j in zip(rows, cols) if valid[i, j]}


def pose_hand_raise(points, min_conf=.4) -> float:
    """A visible single wrist above its shoulder is a cue, not a trained label."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (17, 3) or not np.isfinite(pts).all():
        return 0.
    if min(pts[5, 2], pts[6, 2]) < min_conf:
        return 0.
    shoulder_span = np.linalg.norm(pts[5, :2]-pts[6, :2])
    if shoulder_span < 8:
        return 0.
    strengths = []
    for wrist, shoulder in [(9, 5), (10, 6)]:
        if pts[wrist, 2] >= min_conf and pts[wrist, 1] < pts[shoulder, 1] - .3 * shoulder_span:
            strengths.append(min(pts[wrist, 2], pts[shoulder, 2]))
    return float(max(strengths, default=0.))


def keypoint_quality(points) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (17, 3) or not np.isfinite(pts).all():
        return 0.
    # Lower legs hidden by desks are not counted as missing upper-body evidence.
    return float(np.mean(pts[:11, 2] >= .4))


# These values are documented policy priors, never advertised as learned accuracy.
CONTEXT_SCORES = {
    'unspecified': {'read': .80, 'write': .85, 'raise_hand': .95},
    'lecture': {'read': .75, 'write': .85, 'raise_hand': .95},
    'exercise': {'read': .90, 'write': .95, 'raise_hand': .85},
    'discussion': {'read': .65, 'write': .70, 'raise_hand': .85},
    'presentation': {'read': .65, 'write': .75, 'raise_hand': .90},
}


class ContextTimeline:
    """Explicit teacher/speech/PPT events. Never infer a topic from frame index."""
    def __init__(self, segments=None, default_task='unspecified'):
        if default_task not in CONTEXT_SCORES:
            raise ValueError(f'Unsupported task: {default_task}')
        self.default_task = default_task
        self.segments = sorted(segments or [], key=lambda x: x['start_sec'])
        previous_end = -math.inf
        for segment in self.segments:
            start, end = float(segment['start_sec']), float(segment['end_sec'])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end) or start < previous_end:
                raise ValueError('Context segments must have finite, non-overlapping start/end seconds')
            if segment.get('task', default_task) not in CONTEXT_SCORES:
                raise ValueError('Unsupported context task')
            previous_end = end

    def at(self, timestamp: float) -> dict:
        for segment in self.segments:
            if segment['start_sec'] <= timestamp < segment['end_sec']:
                return {'task': self.default_task, **segment}
        return {'task': self.default_task, 'knowledge_point': ''}


@dataclass
class Evidence:
    time: float
    label: str
    quality: float
    score: float | None
    context: str


class TemporalEvidence:
    """Integrate durations, not frame counts; never bridge long missing gaps."""
    def __init__(self, window_sec=4., max_gap_sec=1.2, min_evidence_sec=.6,
                 retention_sec=6., min_quality=.25):
        if min(window_sec, max_gap_sec, min_evidence_sec, retention_sec) <= 0:
            raise ValueError('Window and retention seconds must be positive')
        self.window_sec = float(window_sec)
        self.max_gap_sec = float(max_gap_sec)
        self.min_evidence_sec = float(min_evidence_sec)
        self.retention_sec = float(retention_sec)
        self.min_quality = float(min_quality)
        self.tracks: dict[int, deque[Evidence]] = {}

    def push(self, track_id, timestamp, label, quality, score, context):
        if not math.isfinite(timestamp):
            raise ValueError('timestamp must be finite')
        history = self.tracks.setdefault(track_id, deque())
        if history and timestamp <= history[-1].time:
            raise ValueError('timestamp must increase within a track')
        history.append(Evidence(timestamp, label, quality, score, context))
        # Keep one sample before the boundary to integrate the crossing interval.
        while len(history) > 1 and history[1].time <= timestamp-self.window_sec:
            history.popleft()

    def prune(self, timestamp):
        for tid in list(self.tracks):
            if timestamp-self.tracks[tid][-1].time > self.retention_sec:
                del self.tracks[tid]

    def assess(self, track_id, timestamp, context):
        report = ObservationAssessment(context=context)
        history = list(self.tracks.get(track_id, []))
        if not history or history[-1].score is None or timestamp-history[-1].time > self.max_gap_sec:
            report.reasons = ['No usable behavior evidence is currently available.']
            return None, report
        start = max(history[0].time, timestamp-self.window_sec)
        span = timestamp-start
        support = weighted = numerator = 0.
        for previous, following in zip(history, history[1:]):
            if previous.context != context or following.context != context:
                continue
            # Gaps and unknown samples remain uncovered, never interpreted as off-task.
            end = min(following.time, previous.time+self.max_gap_sec, timestamp)
            duration = max(0., end-max(previous.time, start))
            if previous.score is not None and previous.quality >= self.min_quality:
                support += duration
                weight = duration * previous.quality
                weighted += weight
                numerator += weight * previous.score
        report.temporal_coverage = round(min(1., support/max(span, 1e-8)), 4)
        report.quality = round(min(1., weighted/max(span, 1e-8)), 4)
        if support + 1e-7 < self.min_evidence_sec or report.quality < self.min_quality or not weighted:
            report.reasons = ['Collecting continuous behavior evidence, or observation quality is insufficient.']
            return None, report
        report.status = 'estimated'
        report.reasons = ['Estimated from visible behavior and the teaching task; not calibrated against classroom engagement labels.',
                          'Unknown behavior is not treated as low engagement. Check coverage alongside the estimate.']
        if context == 'unspecified':
            report.reasons.append('No teaching task was provided; only general participation cue rules are used.')
        return round(numerator/weighted, 4), report


def observation_metrics(students, untracked_people=0):
    known = [s.cas for s in students if s.cas is not None]
    total = len(students) + untracked_people
    return ClassroomMetrics(
        ctes_score=None,
        mean_cas=round(float(np.mean(known)), 4) if known else None,
        std_cas=round(float(np.std(known)), 4) if known else None,
        active_tracks=len(students), known_tracks=len(known), untracked_people=untracked_people,
        coverage=round(len(known)/total, 4) if total else 0.,
        score_kind='rule_based' if known else 'unavailable',
        behavior_distribution=BehaviorDistribution(counts=dict(Counter(s.action.label for s in students))),
    )
