"""Skeleton graph topology for ST-GCN.

Defines the adjacency structure of human body keypoints
for spatial graph convolution.
"""

from __future__ import annotations

import numpy as np


# ── COCO 17-keypoint edge definitions ────────────────────────

COCO_EDGES = [
    (0, 1), (0, 2),       # nose → eyes
    (1, 3), (2, 4),       # eyes → ears
    (0, 5), (0, 6),       # nose → shoulders (via neck approximation)
    (5, 7), (7, 9),       # left arm: shoulder → elbow → wrist
    (6, 8), (8, 10),      # right arm: shoulder → elbow → wrist
    (5, 11), (6, 12),     # shoulders → hips
    (11, 13), (13, 15),   # left leg: hip → knee → ankle
    (12, 14), (14, 16),   # right leg: hip → knee → ankle
    (11, 12),             # hip-hip
    (5, 6),               # shoulder-shoulder
]


class Graph:
    """Skeleton graph for spatial graph convolution.

    Parameters
    ----------
    layout : str
        Keypoint layout (currently only ``'coco'`` is supported).
    strategy : str
        Partition strategy: ``'uniform'``, ``'distance'``, or ``'spatial'``.
    """

    def __init__(self, layout: str = "coco", strategy: str = "spatial") -> None:
        self.num_nodes = 17
        self.edges = list(COCO_EDGES)
        self.center = 0  # nose as center node

        self.A = self._build_adjacency(strategy)

    def _build_adjacency(self, strategy: str) -> np.ndarray:
        """Return adjacency matrices stacked along axis-0.

        For the *spatial* strategy this returns a (3, V, V) tensor
        with three partitions: identity, centripetal, centrifugal.
        """
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i, j in self.edges:
            A[i, j] = 1.0
            A[j, i] = 1.0

        if strategy == "uniform":
            A_with_self = A + np.eye(self.num_nodes, dtype=np.float32)
            D = np.diag(1.0 / np.maximum(A_with_self.sum(axis=1), 1e-6))
            return (D @ A_with_self)[np.newaxis]  # (1, V, V)

        if strategy == "distance":
            # Two partitions: self-loops + neighbours
            I = np.eye(self.num_nodes, dtype=np.float32)
            return np.stack([I, self._normalize(A)], axis=0)  # (2, V, V)

        # strategy == "spatial"  (default)
        hop = self._shortest_path(A)
        I = np.eye(self.num_nodes, dtype=np.float32)
        inward = np.zeros_like(A)
        outward = np.zeros_like(A)

        for i, j in self.edges:
            if hop[j, self.center] < hop[i, self.center]:
                inward[i, j] = 1.0
            else:
                outward[i, j] = 1.0
            if hop[i, self.center] < hop[j, self.center]:
                inward[j, i] = 1.0
            else:
                outward[j, i] = 1.0

        return np.stack(
            [self._normalize(I), self._normalize(inward), self._normalize(outward)],
            axis=0,
        )  # (3, V, V)

    @staticmethod
    def _normalize(A: np.ndarray) -> np.ndarray:
        row_sum = A.sum(axis=1)
        row_sum[row_sum == 0] = 1.0
        return A / row_sum[:, np.newaxis]

    @staticmethod
    def _shortest_path(A: np.ndarray) -> np.ndarray:
        """Floyd-Warshall shortest-path matrix."""
        n = A.shape[0]
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        for i in range(n):
            for j in range(n):
                if A[i, j] > 0:
                    dist[i, j] = 1
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist
