"""
DeepSORT多目标跟踪封装
"""
import numpy as np
from deep_sort import DeepSort


class MultiObjectTracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, bboxes, features, frame_size):
        """
        更新跟踪器

        Args:
            bboxes: 检测框 [[x1,y1,x2,y2], ...]
            features: 特征向量
            frame_size: (width, height)

        Returns:
            跟踪结果 [[x1,y1,x2,y2,track_id], ...]
        """
        if len(bboxes) == 0:
            return []

        return self.tracker.update(bboxes, features, frame_size)


# 备用tracker（无DeepSort时使用）
class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def update(self, bboxes, *args):
        tracks = []
        for i, bbox in enumerate(bboxes):
            tracks.append([*bbox, i])
        return tracks