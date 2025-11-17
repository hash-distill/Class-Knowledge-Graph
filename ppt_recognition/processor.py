import cv2
import numpy as np
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger

# 延迟导入paddleocr，避免未安装时报错
try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.warning("PaddleOCR未安装，PPT识别功能将不可用")


@dataclass
class SlideInfo:
    """PPT识别结果"""
    timestamp: float
    knowledge_point: str
    raw_text: str
    confidence: float
    bbox: Optional[list] = None


class PPTProcessor:
    """
    PPT文字识别处理器
    功能：从视频流/屏幕中检测PPT区域并提取知识点
    """

    def __init__(self, gpu_id: int = 0, config: dict = None):
        """
        初始化处理器

        Args:
            gpu_id: GPU设备ID
            config: 配置字典
        """
        self.config = config or {}
        self.gpu_id = gpu_id

        # 初始化OCR引擎
        try:
            self.ocr = PaddleOCR(
                use_gpu=self.config.get("use_gpu", True),
                gpu_id=gpu_id,
                lang=self.config.get("language", "ch"),
                det_db_thresh=self.config.get("det_db_thresh", 0.3),
                det_db_box_thresh=self.config.get("det_db_box_thresh", 0.5),
                enable_mkldnn=True,
                det_limit_side_len=960
            )
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            self.ocr = None

        # 状态变量
        self.last_fingerprint = None
        self.stable_count = 0
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.simhash_threshold = self.config.get("simhash_threshold", 5)
        self.stable_frame_count = self.config.get("stable_frame_count", 30)

        # 知识点提取模式
        self.patterns = {
            'definition': r'(定义|概念|含义):?\s*([^。\n]+)',
            'theorem': r'(定理|定律|法则|原理):?\s*([^。\n]+)',
            'formula': r'(公式|方程|表达式):?\s*([^。\n]+)',
            'example': r'(例[题子]|案例|示例):?\s*([^。\n]+)',
            'conclusion': r'(结论|总结|归纳):?\s*([^。\n]+)'
        }


    def detect_slide_region(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        检测PPT区域

        Args:
            frame: 输入帧

        Returns:
            tuple: (处理后的图像, 四边形顶点坐标)
        """
        try:
            # 灰度化和高斯模糊
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny边缘检测
            edges = cv2.Canny(blur, 50, 150)

            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return frame, None

            # 筛选四边形轮廓
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(cnt)
                if area < frame.shape[0] * frame.shape[1] * 0.1:
                    continue

                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4:
                    # 绘制检测框（调试用）
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    return frame, approx.reshape(4, 2)

            return frame, None

        except Exception as e:
            logger.error(f"PPT区域检测失败: {e}")
            return frame, None

    def perspective_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        四点透视矫正

        Args:
            image: 原始图像
            pts: 四边形顶点

        Returns:
            矫正后的图像
        """
        try:
            rect = self.order_points(pts)
            (tl, tr, br, bl) = rect

            # 计算目标尺寸
            width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

            # 目标矩形
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

            # 透视变换矩阵
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (width, height))

            return warped

        except Exception as e:
            logger.error(f"透视变换失败: {e}")
            return image

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        四点排序：左上→右上→右下→左下

        Args:
            pts: 原始四边形顶点

        Returns:
            排序后的顶点
        """
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下

        return rect

    def compute_fingerprint(self, text: str) -> int:
        """
        SimHash文本指纹计算，用于去重

        Args:
            text: 输入文本

        Returns:
            64位指纹整数
        """
        # 限制文本长度
        words = text[:200].split()
        if not words:
            return 0

        v = np.zeros(64)
        for word in words:
            h = hash(word) & ((1 << 64) - 1)
            for i in range(64):
                v[i] += 1 if h & (1 << i) else -1

        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    def extract_knowledge(self, text: str) -> str:
        """
        从OCR文本提取核心知识点

        Args:
            text: 完整OCR文本

        Returns:
            结构化知识点字符串
        """
        # 先尝试匹配模式
        for key, pattern in self.patterns.items():
            match = re.search(pattern, text)
            if match:
                return f"{key}: {match.group(2).strip()}"

        # 提取标题行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if 5 < len(line) < 30:
                return f"topic: {line}"

        # 兜底：返回前50字符
        return f"content: {text[:50]}"

    def process_frame(self, frame: np.ndarray) -> Optional[SlideInfo]:
        """
        处理单帧图像

        Args:
            frame: 输入帧

        Returns:
            SlideInfo对象或None
        """
        try:
            # 1. 检测并矫正PPT区域
            processed_frame, pts = self.detect_slide_region(frame)
            if pts is not None:
                slide_img = self.perspective_transform(processed_frame, pts)
            else:
                slide_img = frame

            # 2. OCR识别
            if self.ocr is None:
                logger.error("OCR引擎未初始化")
                return None

            result = self.ocr.ocr(slide_img, cls=True)
            if not result or not result[0]:
                return None

            # 3. 文本整合与过滤
            full_text = "\n".join([
                line[1][0] for line in result[0]
                if line[1][1] > self.confidence_threshold
            ])

            if not full_text:
                return None

            # 4. 去重判断
            fingerprint = self.compute_fingerprint(full_text)
            is_duplicate = False

            if self.last_fingerprint is not None:
                diff_bits = bin(fingerprint ^ self.last_fingerprint).count('1')
                if diff_bits < self.simhash_threshold:
                    self.stable_count += 1
                    if self.stable_count < self.stable_frame_count:
                        return None  # 内容未稳定
                    is_duplicate = True

            # 更新状态
            self.last_fingerprint = fingerprint
            self.stable_count = 0

            # 5. 知识点提取
            knowledge = self.extract_knowledge(full_text)

            # 6. 构建结果
            slide_info = SlideInfo(
                timestamp=time.time(),
                knowledge_point=knowledge,
                raw_text=full_text,
                confidence=result[0][0][1][1] if result[0] else 0.0,
                bbox=result[0][0][0] if result[0] else None
            )

            return slide_info

        except Exception as e:
            logger.error(f"帧处理失败: {e}")
            return None