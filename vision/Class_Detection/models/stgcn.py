"""ST-GCN: Spatial-Temporal Graph Convolutional Network.

Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition", AAAI 2018.

Adapted for classroom action recognition with 17 COCO keypoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.graph import Graph


class STGCNBlock(nn.Module):
    """A single ST-GCN block: spatial GCN + temporal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()

        num_partitions = A.shape[0]

        # Spatial graph convolution
        self.gcn = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1)
        self.A = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=False
        )
        self.num_partitions = num_partitions
        self.out_channels = out_channels

        # Learnable edge-importance weighting
        self.edge_importance = nn.Parameter(torch.ones_like(self.A))

        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                      stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, C_in, T, V)
        """
        res = self.residual(x)

        # Spatial GCN
        N, C, T, V = x.shape
        h = self.gcn(x)  # (N, C_out * P, T, V)
        h = h.view(N, self.num_partitions, self.out_channels, T, V)

        # Aggregate across partitions with edge importance
        A_weighted = self.A * self.edge_importance
        out = torch.zeros(N, self.out_channels, T, V, device=x.device)
        for p in range(self.num_partitions):
            out += torch.einsum("nctv,vw->nctw", h[:, p], A_weighted[p])

        # Temporal convolution + residual
        out = self.tcn(out) + res
        return self.relu(out)


class STGCN(nn.Module):
    """Full ST-GCN model for skeleton-based action recognition.

    Input shape:  (N, C=3, T=30, V=17, M=1)
    Output shape: (N, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        graph: Graph | None = None,
        edge_importance: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if graph is None:
            graph = Graph(layout="coco", strategy="spatial")
        A = graph.A  # (P, V, V)

        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[2])

        # ST-GCN blocks
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, stride=1, dropout=dropout, residual=False),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 128, A, stride=2, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout),
            STGCNBlock(256, 256, A, stride=1, dropout=dropout),
            STGCNBlock(256, 256, A, stride=1, dropout=dropout),
        ])

        if not edge_importance:
            for block in self.blocks:
                block.edge_importance.requires_grad_(False)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (N, C, T, V, M)
            C=3 (x, y, confidence), T=window, V=17 keypoints, M=1 person.
        """
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(N, M, -1).mean(dim=1)

        return self.fc(x)
