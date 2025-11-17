import cv2
import mediapipe as mp
import torch
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Optional
import threading
from loguru import logger

from student_engagement.models.multitask_net import SAGENet

try:
    from deep_sort import DeepSort
except ImportError:
    logger.error("DeepSort未安装，多目标跟踪功能将不可用")


class SAGEAnalyzer:
    """
    SAGE-Net分析引擎
    功能：分析学生课堂状态，输出参与度评分
    """

    def __init__(self, model_path: str, device: str = 'cuda', config: dict = None):
        """
        初始化分析引擎

        Args:
            model_path: SAGE-Net模型权重路径
            device: 推理设备
            config: 配置字典
        """
        self.config = config or {}
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')


        # 加载SAGE-Net模型
        try:
            self.model = SAGENet(
                expression_classes=self.config.get("expression_classes", 7)
            ).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict']
                                       if 'model_state_dict' in checkpoint
                                       else checkpoint)
            self.model.eval()
        except Exception as e:
            logger.error(f"SAGE-Net模型加载失败: {e}")
            raise

        # MediaPipe工具初始化
        self._init_mediapipe()

        # 多目标跟踪器
        self.tracker = DeepSort(
            max_age=self.config.get("max_age", 30),
            n_init=self.config.get("n_init", 3)
        )

        # 状态缓存
        self.student_states = defaultdict(lambda: {
            "engagement_buffer": [],
            "action_history": [],
            "last_seen": 0,
            "total_score": 0
        })

        # 情绪标签映射
        self.emotions = ["专注", "困惑", "疲惫", "积极", "中性", "消极", "分心"]

        # 性能统计
        self.stats = {
            "total_frames": 0,
            "avg_inference_time": 0.0,
            "face_detect_count": 0
        }


    def _init_mediapipe(self):
        """初始化MediaPipe工具"""
        try:
            # 人脸检测
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.config.get("face_confidence", 0.7)
            )

            # 人脸网格（用于头部姿态）
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.config.get("max_num_faces", 10),
                min_detection_confidence=0.5
            )

            # 姿态检测（用于行为识别）
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5
            )

        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {e}")
            raise

    def preprocess_face(self, face_crop: np.ndarray, size: int = 224) -> torch.Tensor:
        """
        人脸图像预处理

        Args:
            face_crop: 裁剪的人脸图像
            size: 目标尺寸

        Returns:
            归一化后的张量
        """
        try:
            # 调整尺寸
            face_crop = cv2.resize(face_crop, (size, size))

            # BGR→RGB
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # 归一化
            face_crop = face_crop.astype(np.float32) / 255.0

            # HWC→CHW
            face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1)

            return face_tensor

        except Exception as e:
            logger.error(f"人脸预处理失败: {e}")
            # 返回空张量
            return torch.zeros(3, size, size)

    def analyze_frame(self, frame: np.ndarray) -> Dict[int, dict]:
        """
        分析单帧图像中的多学生状态

        Args:
            frame: 输入帧

        Returns:
            学生ID到状态信息的字典
        """
        start_time = time.perf_counter()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # 1. 人脸检测
        face_results = self.mp_face.process(rgb_frame)
        if not face_results.detections:
            self.stats["total_frames"] += 1
            return {}

        detection_count = len(face_results.detections)
        self.stats["face_detect_count"] += detection_count

        # 2. 提取检测框和人脸图像
        bboxes, face_crops = [], []
        for det in face_results.detections:
            bbox = self._get_bbox(det, w, h)
            face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            if face_crop.size > 0 and face_crop.shape[0] > 30:
                bboxes.append(bbox)
                face_crops.append(self.preprocess_face(face_crop))

        if not bboxes:
            self.stats["total_frames"] += 1
            return {}

        # 3. 多目标跟踪（保持ID一致性）
        try:
            # 生成特征（这里使用随机特征，实际应使用ReID模型）
            features = torch.rand(len(bboxes), 128).numpy()
            tracks = self.tracker.update(np.array(bboxes), features, (w, h))
        except Exception as e:
            logger.error(f"多目标跟踪失败: {e}")
            # 使用简单ID分配
            tracks = [[0, 0, 0, 0, i] for i in range(len(bboxes))]

        # 4. SAGE-Net批量推理
        try:
            batch_tensor = torch.stack(face_crops).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
        except Exception as e:
            logger.error(f"SAGE-Net推理失败: {e}")
            self.stats["total_frames"] += 1
            return {}

        # 5. 结果组装
        results = {}
        for i, track in enumerate(tracks):
            try:
                track_id = int(track[-1])

                # 解析SAGE-Net输出
                head_pose = outputs["head_pose"][i].cpu().numpy()
                gaze = outputs["gaze"][i].cpu().numpy()
                expression_idx = torch.argmax(outputs["expression"][i]).item()
                engagement_score = outputs["engagement"][i].item() * 100

                # 判断注意力状态
                attention_status = self._judge_attention(gaze, head_pose)

                # 构建结果
                results[track_id] = {
                    "student_id": track_id,
                    "engagement_score": round(engagement_score, 2),
                    "attention": attention_status,
                    "emotion": self.emotions[expression_idx],
                    "emotion_prob": torch.softmax(outputs["expression"][i], dim=0).cpu().numpy().tolist(),
                    "gaze": gaze.tolist(),
                    "head_pose": head_pose.tolist(),
                    "bbox": bboxes[i],
                    "timestamp": time.time()
                }

                # 更新学生状态缓存
                self._update_state(track_id, engagement_score, attention_status)

            except Exception as e:
                logger.error(f"结果组装失败 (track {track_id}): {e}")
                continue

        # 更新性能统计
        inference_time = (time.perf_counter() - start_time) * 1000
        self._update_stats(inference_time)

        return results

    def _get_bbox(self, detection, img_w: int, img_h: int) -> List[int]:
        """
        MediaPipe检测框转换为像素坐标

        Args:
            detection: MediaPipe检测结果
            img_w: 图像宽度
            img_h: 图像高度

        Returns:
            边界框 [x1, y1, x2, y2]
        """
        try:
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * img_w))
            y1 = max(0, int(bbox.ymin * img_h))
            x2 = min(img_w, int((bbox.xmin + bbox.width) * img_w))
            y2 = min(img_h, int((bbox.ymin + bbox.height) * img_h))
            return [x1, y1, x2, y2]
        except Exception as e:
            logger.error(f"边界框转换失败: {e}")
            return [0, 0, 100, 100]

    def _judge_attention(self, gaze: np.ndarray, head_pose: np.ndarray,
                         thresholds: tuple = (0.4, 30)) -> str:
        """
        综合判断学生注意力状态

        Args:
            gaze: 视线方向 [x, y]
            head_pose: 头部姿态 [pitch, yaw, roll]
            thresholds: (视线阈值, 头部偏转阈值)

        Returns:
            状态字符串: focused/distracted/low_engagement
        """
        gaze_x, gaze_y = gaze
        pitch, yaw, roll = head_pose

        # 视线在黑板区域且头部未过度偏转
        if abs(gaze_x) < thresholds[0] and abs(yaw) < thresholds[1]:
            return "focused"
        # 严重偏离或头部转动过大
        elif abs(gaze_x) > 0.7 or abs(yaw) > 60:
            return "distracted"
        else:
            return "low_engagement"

    def _update_state(self, student_id: int, engagement: float, status: str):
        """
        更新学生状态缓存

        Args:
            student_id: 学生ID
            engagement: 参与度评分
            status: 注意力状态
        """
        state = self.student_states[student_id]
        state["engagement_buffer"].append(engagement)
        state["action_history"].append(status)
        state["last_seen"] = time.time()

        # 限制缓存大小
        max_buffer = self.config.get("engagement_buffer_size", 50)
        if len(state["engagement_buffer"]) > max_buffer:
            state["engagement_buffer"].pop(0)
            state["action_history"].pop(0)

        # 更新平均分数
        state["total_score"] = np.mean(state["engagement_buffer"])

    def _update_stats(self, inference_time: float):
        """更新性能统计"""
        self.stats["total_frames"] += 1
        alpha = 0.9
        self.stats["avg_inference_time"] = (
                alpha * self.stats["avg_inference_time"] +
                (1 - alpha) * inference_time
        )

    def get_stats(self) -> dict:
        """获取性能统计"""
        return {
            **self.stats,
            "avg_engagement": np.mean([
                np.mean(state["engagement_buffer"])
                for state in self.student_states.values()
                if state["engagement_buffer"]
            ]) if self.student_states else 0,
            "active_students": len(self.student_states)
        }