"""Classroom evaluation CV pipeline.

Public re-exports for convenient imports::

    from src import ClassroomPipeline, Detector, PoseEstimator

Heavy dependencies (ultralytics, paddleocr) are lazily imported so that
lightweight modules (scoring, vsam, schema) can be used without GPU
packages installed.
"""

# ── Always-available (no heavy deps) ─────────────────────────
from src.schema import (
    ClassroomSnapshot,
    StudentState,
    ClassroomMetrics,
    KnowledgeAnchor,
    BBoxRecord,
    KeypointRecord,
    ActionRecord,
    GazeRecord,
)
from src.scoring import calc_cas, calc_ctes, compute_classroom_metrics
from src.vsam import VSAMAligner, gaussian_weight, score_knowledge_point
from src.gaze import GazeEstimator


def __getattr__(name: str):
    """Lazy import for modules that require ultralytics / paddleocr."""
    if name == "Detector":
        from src.detector import Detector
        return Detector
    if name == "PoseEstimator":
        from src.pose import PoseEstimator
        return PoseEstimator
    if name == "ActionClassifier":
        from src.action import ActionClassifier
        return ActionClassifier
    if name == "OCRAnchorDetector":
        from src.ocr_anchor import OCRAnchorDetector
        return OCRAnchorDetector
    if name == "ClassroomPipeline":
        from src.pipeline import ClassroomPipeline
        return ClassroomPipeline
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    # Pipeline (lazy)
    "ClassroomPipeline",
    # Modules (lazy)
    "Detector",
    "PoseEstimator",
    "ActionClassifier",
    "OCRAnchorDetector",
    # Modules (eager)
    "GazeEstimator",
    "VSAMAligner",
    # Scoring (eager)
    "calc_cas",
    "calc_ctes",
    "compute_classroom_metrics",
    "gaussian_weight",
    "score_knowledge_point",
    # Data models (eager)
    "ClassroomSnapshot",
    "StudentState",
    "ClassroomMetrics",
    "KnowledgeAnchor",
    "BBoxRecord",
    "KeypointRecord",
    "ActionRecord",
    "GazeRecord",
]
