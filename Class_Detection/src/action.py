"""Action classification from keypoint sequences.

Two paths:
  1. **ST-GCN** (primary) – temporal graph convolution on keypoint windows.
  2. **Rule-based** (fallback) – heuristic scoring from detection class names.

The pipeline starts with rule-based classification by default and
switches to ST-GCN once a trained checkpoint is available.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.schema import ActionRecord, ActionSource


# ── Action label mapping ──────────────────────────────────────

# Maps detection class names (from SCB-Dataset5) to engagement scores.
# This table is the fallback when ST-GCN is not available.
DEFAULT_ACTION_SCORES: dict[str, float] = {
    # 学生行为 (SCB-5 unified 13-class system)
    "hand_raising": 0.95,
    "read": 0.72,
    "write": 0.78,
    "discuss": 0.82,
    "talk": 0.75,
    "answer": 0.90,
    "stage_interact": 0.88,
    "stand": 0.65,
    # 教师/环境 (不计入学生评分)
    "teacher": 0.0,
    "guide": 0.0,
    "board_writing": 0.0,
    "blackboard": 0.0,
    "screen": 0.0,
}

# ST-GCN output labels (index → name)
STGCN_LABELS: dict[int, str] = {
    0: "writing",
    1: "reading",
    2: "hand_raising",
    3: "discussing",
    4: "attending",
    5: "leaning",
    6: "using_phone",
    7: "yawning",
    8: "looking_around",
}

STGCN_ENGAGEMENT: dict[str, float] = {
    "writing": 0.85,
    "reading": 0.80,
    "hand_raising": 0.95,
    "discussing": 0.70,
    "attending": 0.70,
    "leaning": 0.15,
    "using_phone": 0.10,
    "yawning": 0.20,
    "looking_around": 0.25,
}


# ── Keypoint buffer (per-track) ──────────────────────────────

class KeypointBuffer:
    """Maintains a fixed-size sliding window of keypoints per track ID."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._buffers: dict[int, deque] = {}

    def push(self, track_id: int, keypoints: list[list[float]]) -> None:
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)
        self._buffers[track_id].append(keypoints)

    def get_window(self, track_id: int) -> Optional[np.ndarray]:
        """Return (C, T, V, 1) tensor if enough frames are buffered."""
        buf = self._buffers.get(track_id)
        if buf is None or len(buf) < self.window_size:
            return None
        # Stack: (T, V, C) → transpose to (C, T, V) → add M dim
        arr = np.array(list(buf), dtype=np.float32)  # (T, 17, 3)
        arr = arr.transpose(2, 0, 1)  # (3, T, 17)
        return arr[:, :, :, np.newaxis]  # (3, T, 17, 1)

    def clear(self, track_id: int) -> None:
        self._buffers.pop(track_id, None)

    def active_ids(self) -> set[int]:
        return set(self._buffers.keys())

    def prune(self, keep_ids: set[int]) -> None:
        """Remove buffers for track IDs not in *keep_ids*."""
        stale = [tid for tid in self._buffers if tid not in keep_ids]
        for tid in stale:
            del self._buffers[tid]


# ── Action classifier ────────────────────────────────────────

class ActionClassifier:
    """Classify student actions via ST-GCN or rule-based fallback."""

    def __init__(
        self,
        use_stgcn: bool = False,
        stgcn_weights: Optional[str | Path] = None,
        window_size: int = 30,
        action_scores: Optional[dict[str, float]] = None,
        device: str = "0",
    ) -> None:
        self.use_stgcn = use_stgcn
        self.window_size = window_size
        self.action_scores = action_scores or DEFAULT_ACTION_SCORES
        self.device = device
        self.buffer = KeypointBuffer(window_size=window_size)

        self._stgcn_model: Optional[torch.nn.Module] = None
        if use_stgcn and stgcn_weights:
            self._load_stgcn(stgcn_weights)

    # ── public API ────────────────────────────────────────────

    def classify_from_detection(
        self, class_name: str, confidence: float
    ) -> ActionRecord:
        """Quick classification using the detection class name directly.

        The *engagement_score* (how engaged the action is) is kept separate
        from *confidence* (how certain the detector is about the class).
        CAS should use engagement_score, not the product of the two.
        """
        # If the generic detector just outputs "person", map it to a neutral "attending" state
        if class_name.lower() == "person":
            label = "attending"
            engagement = 0.70
        else:
            label = class_name
            engagement = self.action_scores.get(class_name, 0.5)

        return ActionRecord(
            label=label,
            confidence=round(confidence, 4),
            engagement_score=round(engagement, 4),
            det_confidence=round(confidence, 4),
            source=ActionSource.DETECTION,
        )

    def classify_from_keypoints(self, track_id: int) -> Optional[ActionRecord]:
        """Classify using buffered keypoint window (ST-GCN or rule)."""
        window = self.buffer.get_window(track_id)
        if window is None:
            return None

        if self.use_stgcn and self._stgcn_model is not None:
            return self._stgcn_infer(window)

        return self._rule_infer(window)

    def push_keypoints(self, track_id: int, keypoints: list[list[float]]) -> None:
        """Add a frame's keypoints to the buffer for *track_id*."""
        self.buffer.push(track_id, keypoints)

    # ── ST-GCN ────────────────────────────────────────────────

    def _load_stgcn(self, weights_path: str | Path) -> None:
        from models.stgcn import STGCN
        from models.graph import Graph

        # Resolve target device
        if self.device.isdigit() and torch.cuda.is_available():
            self._torch_device = torch.device(f"cuda:{self.device}")
        else:
            self._torch_device = torch.device("cpu")

        graph = Graph(layout="coco", strategy="spatial")
        model = STGCN(
            in_channels=3,
            num_classes=len(STGCN_LABELS),
            graph=graph,
            edge_importance=True,
            dropout=0.3,
        )
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        model = model.to(self._torch_device)
        self._stgcn_model = model

    @torch.no_grad()
    def _stgcn_infer(self, window: np.ndarray) -> ActionRecord:
        device = getattr(self, "_torch_device", torch.device("cpu"))
        tensor = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, C, T, V, M)
        logits = self._stgcn_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        idx = int(probs.argmax().item())
        label = STGCN_LABELS.get(idx, "unknown")
        conf = float(probs[idx].item())
        engagement = STGCN_ENGAGEMENT.get(label, 0.5)
        return ActionRecord(
            label=label,
            confidence=round(conf, 4),
            engagement_score=round(engagement, 4),
            source=ActionSource.STGCN,
        )

    # ── Rule-based fallback ──────────────────────────────────

    @staticmethod
    def _rule_infer(window: np.ndarray) -> ActionRecord:
        """Simple heuristic from keypoint motion statistics.

        All thresholds are normalised by the median torso height
        (shoulder-to-hip distance) so the rules are resolution-independent.
        """
        # window shape: (3, T, 17, 1)
        coords = window[:2, :, :, 0]  # (2, T, 17)

        # Compute normalisation scale: median shoulder-to-hip distance
        shoulders_y = coords[1, :, 5:7].mean(axis=1)       # (T,)
        hips_y = coords[1, :, 11:13].mean(axis=1)          # (T,)
        torso_height = float(np.median(np.abs(hips_y - shoulders_y)))
        scale = max(torso_height, 1.0)  # avoid division by zero

        # Head motion range (keypoints 0..4), normalised
        head = coords[:, :, :5]  # (2, T, 5)
        head_range = float(np.mean(np.ptp(head, axis=1))) / scale

        # Hand height relative to shoulders (keypoints 9,10 vs 5,6)
        wrists_y = coords[1, :, 9:11].mean(axis=1)         # (T,)
        hand_raised_ratio = float(np.mean(wrists_y < shoulders_y - 0.15 * scale))

        # Body lean: nose y relative to shoulder y
        nose_y = coords[1, :, 0]                            # (T,)
        lean_ratio = float(np.mean(nose_y > shoulders_y + 0.25 * scale))

        if hand_raised_ratio > 0.5:
            return ActionRecord(label="hand_raising", confidence=0.85, engagement_score=0.95, source=ActionSource.RULE)
        if lean_ratio > 0.5:
            return ActionRecord(label="leaning", confidence=0.75, engagement_score=0.15, source=ActionSource.RULE)
        if head_range > 0.20:
            return ActionRecord(label="looking_around", confidence=0.65, engagement_score=0.25, source=ActionSource.RULE)
        if head_range < 0.05:
            return ActionRecord(label="attending", confidence=0.60, engagement_score=0.70, source=ActionSource.RULE)

        return ActionRecord(label="attending", confidence=0.50, engagement_score=0.60, source=ActionSource.RULE)
