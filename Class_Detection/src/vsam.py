"""VSAM: Visual-Semantic Alignment Model.

Implements Gaussian-prior temporal alignment between OCR-detected
knowledge-point anchors and student engagement signals (CAS values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ── Gaussian weight ──────────────────────────────────────────

def gaussian_weight(
    t: float, t_ocr: float, mu: float = 3.0, sigma: float = 1.5
) -> float:
    """Gaussian reaction-delay weight centered at (t_ocr + mu).

    Parameters
    ----------
    t : float
        Current timestamp (seconds).
    t_ocr : float
        Time when the knowledge point was detected by OCR.
    mu : float
        Expected reaction delay (seconds).
    sigma : float
        Width of the Gaussian window.
    """
    sigma = max(float(sigma), 1e-6)
    center = float(t_ocr) + float(mu)
    return math.exp(-((float(t) - center) ** 2) / (2.0 * sigma * sigma))


# ── Knowledge-point absorption score ─────────────────────────

def score_knowledge_point(
    cas_values: list[float],
    timestamps: list[float],
    t_ocr: float,
    mu: float = 3.0,
    sigma: float = 1.5,
) -> float:
    """Weighted average of CAS values using Gaussian alignment.

    $$Score_{K_i} = \\frac{\\sum W(t) \\cdot CAS(t)}{\\sum W(t)}$$
    """
    if not cas_values or not timestamps or len(cas_values) != len(timestamps):
        return 0.0

    weights = [gaussian_weight(t, t_ocr, mu, sigma) for t in timestamps]
    denom = sum(weights)
    if denom <= 1e-12:
        return 0.0

    numer = sum(w * float(s) for w, s in zip(weights, cas_values))
    return _clamp01(numer / denom)


# ── Anchor manager ───────────────────────────────────────────

@dataclass
class KnowledgeAnchorEvent:
    """A single OCR-triggered knowledge-point event."""
    entity: str
    t_ocr: float
    cas_buffer: list[float] = field(default_factory=list)
    time_buffer: list[float] = field(default_factory=list)
    score_k: float = 0.0
    closed: bool = False


class VSAMAligner:
    """Manages knowledge-point anchors and soft-alignment scoring.

    Lifecycle:
      1. ``trigger()``   → OCR detects a new knowledge point
      2. ``feed()``      → per-frame CAS values are appended
      3. ``evaluate()``  → compute Score_Ki for mature anchors
    """

    def __init__(
        self,
        mu: float = 3.0,
        sigma: float = 1.5,
        window_duration: float = 12.0,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.window_duration = window_duration
        self._anchors: list[KnowledgeAnchorEvent] = []

    @property
    def active_anchors(self) -> list[KnowledgeAnchorEvent]:
        return [a for a in self._anchors if not a.closed]

    @property
    def all_anchors(self) -> list[KnowledgeAnchorEvent]:
        return list(self._anchors)

    def trigger(self, entity: str, t_ocr: float) -> KnowledgeAnchorEvent:
        """Register a new knowledge-point anchor (called by OCR module)."""
        anchor = KnowledgeAnchorEvent(entity=entity, t_ocr=t_ocr)
        self._anchors.append(anchor)
        return anchor

    def feed(self, t: float, mean_cas: float) -> None:
        """Append a CAS observation to all active anchors."""
        for anchor in self.active_anchors:
            if t - anchor.t_ocr <= self.window_duration:
                anchor.cas_buffer.append(mean_cas)
                anchor.time_buffer.append(t)

    def evaluate(self, current_time: float) -> list[KnowledgeAnchorEvent]:
        """Score and close mature anchors whose window has elapsed."""
        newly_closed: list[KnowledgeAnchorEvent] = []
        for anchor in self.active_anchors:
            if current_time - anchor.t_ocr > self.window_duration:
                anchor.score_k = score_knowledge_point(
                    anchor.cas_buffer, anchor.time_buffer,
                    anchor.t_ocr, self.mu, self.sigma,
                )
                anchor.closed = True
                newly_closed.append(anchor)
        return newly_closed
