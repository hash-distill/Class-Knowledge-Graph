"""Pydantic data models for the classroom evaluation pipeline.

All inter-module data flows are typed through these schemas so that
serialization, validation, and documentation stay consistent.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────

class GazeSource(str, Enum):
    PNP = "pnp"
    FALLBACK = "fallback"
    PRIOR = "prior"


class ActionSource(str, Enum):
    STGCN = "stgcn"
    RULE = "rule"
    DETECTION = "detection"


class FocusZone(str, Enum):
    BOARD = "board_focus"
    DESK = "desk_focus"
    WANDERING = "wandering"


# ── Per-object records ────────────────────────────────────────

class BBoxRecord(BaseModel):
    """A single detected object."""
    class_id: int
    class_name: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    xyxy: list[float] = Field(min_length=4, max_length=4)
    track_id: Optional[int] = None


class KeypointRecord(BaseModel):
    """17 COCO keypoints for one person."""
    points: list[list[float]]  # shape (17, 3): [x, y, conf]
    mean_confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class ActionRecord(BaseModel):
    label: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    engagement_score: float = Field(ge=0.0, le=1.0, default=0.5)
    det_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: ActionSource = ActionSource.RULE


class GazeRecord(BaseModel):
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    focus_score: float = Field(ge=0.0, le=1.0, default=0.5)
    focus_zone: FocusZone = FocusZone.WANDERING
    source: GazeSource = GazeSource.PRIOR


# ── Student-level aggregation ─────────────────────────────────

class StudentState(BaseModel):
    track_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    action: ActionRecord = Field(default_factory=ActionRecord)
    gaze: GazeRecord = Field(default_factory=GazeRecord)
    cas: float = Field(ge=0.0, le=1.0, default=0.0)


# ── Knowledge anchor ─────────────────────────────────────────

class KnowledgeAnchor(BaseModel):
    entity: str = ""
    trigger_time: str = ""
    gaussian_weight: float = Field(ge=0.0, le=1.0, default=0.0)
    score_k: float = Field(ge=0.0, le=1.0, default=0.0)
    visual_score: float = Field(ge=0.0, le=1.0, default=0.0)


# ── Classroom-level aggregation ──────────────────────────────

class BehaviorDistribution(BaseModel):
    """Counts of each detected behavior in the current frame."""
    counts: dict[str, int] = Field(default_factory=dict)


class ClassroomMetrics(BaseModel):
    ctes_score: float = Field(ge=0.0, le=1.0, default=0.0)
    mean_cas: float = Field(ge=0.0, le=1.0, default=0.0)
    std_cas: float = Field(ge=0.0, default=0.0)
    active_tracks: int = 0
    behavior_distribution: BehaviorDistribution = Field(
        default_factory=BehaviorDistribution
    )


# ── Top-level snapshot (output per evaluation window) ────────

class ClassroomSnapshot(BaseModel):
    """The complete structured output of one evaluation cycle."""
    timestamp: str
    frame_id: int = 0
    frame_image_path: Optional[str] = None
    knowledge_anchor: KnowledgeAnchor = Field(
        default_factory=KnowledgeAnchor
    )
    classroom_metrics: ClassroomMetrics = Field(
        default_factory=ClassroomMetrics
    )
    student_states: list[StudentState] = Field(default_factory=list)
