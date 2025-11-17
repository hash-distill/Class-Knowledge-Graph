# paddle_ppt_ocr.py
import os
import cv2
import time
import argparse
import numpy as np
import logging
from paddleocr import PaddleOCR
from typing import List, Dict

# 控制日志输出（只显示警告及以上级别）
logging.basicConfig(level=logging.ERROR)


class PPTOCRProcessor:
    def __init__(self, lang: str = "ch"):
        """初始化PaddleOCR模型（严格适配2.7.0版本，移除所有不支持的参数）"""
        self.ocr = PaddleOCR(
            lang=lang  # 仅保留语言参数（2.7.0版本明确支持）
            # 移除 cls、show_log 等不支持的参数
        )
        print(f"PaddleOCR模型加载完成（语言：{lang}，版本：2.6.0）")

    def _parse_ocr_result(self, ocr_output) -> List[Dict]:
        """解析PaddleOCR输出结果"""
        parsed = []
        if not ocr_output or len(ocr_output) == 0:
            return parsed
        # 2.7.0版本输出格式：[[(检测框), (文字, 置信度)], ...]
        for line in ocr_output[0]:  # 取第一页（单张图片）的结果
            box = line[0]  # 检测框坐标：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text, conf = line[1]  # 文字内容和置信度
            # 转换为左上角和右下角坐标
            x1, y1 = min(p[0] for p in box), min(p[1] for p in box)
            x2, y2 = max(p[0] for p in box), max(p[1] for p in box)
            parsed.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "text": text,
                "confidence": round(conf, 3)
            })
        return parsed

    def process_single_image(self, img_path: str, use_cls: bool = False) -> List[Dict]:
        """处理单张图片，use_cls控制是否进行方向分类（在调用时指定，而非初始化）"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片不存在：{img_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片：{img_path}")

        # 调用识别方法时指定是否启用方向分类（cls参数在这里传入，而非初始化）
        result = self.ocr.ocr(img, cls=use_cls)
        return self._parse_ocr_result(result)

    def process_video_frames(self, video_path: str, interval: int = 5, use_cls: bool = False) -> List[Dict]:
        """处理视频帧，use_cls控制方向分类"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_count = 0

        print(f"开始处理视频：{video_path}（帧率：{fps:.1f}，总帧数：{total_frames}）")
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                # 调用时指定cls参数
                ocr_result = self.ocr.ocr(frame, cls=use_cls)
                parsed_result = self._parse_ocr_result(ocr_result)

                results.append({
                    "timestamp": round(timestamp, 2),
                    "frame_idx": frame_count,
                    "ocr_texts": parsed_result
                })

                if frame_count % (frame_interval * 10) == 0:
                    print(f"已处理 {frame_count}/{total_frames} 帧，时间戳：{timestamp:.2f}s")

            frame_count += 1

        cap.release()
        end_time = time.time()
        print(f"视频处理完成，耗时：{end_time - start_time:.2f}s，共提取 {len(results)} 帧结果")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPT文字识别工具（完全适配PaddleOCR 2.7.0）")
    parser.add_argument("--type", type=str, required=True, choices=["image", "video"],
                        help="处理类型：image（单张图片）或 video（视频）")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径（图片或视频）")
    parser.add_argument("--interval", type=int, default=5,
                        help="视频处理时间间隔（秒），默认5秒")
    parser.add_argument("--use-cls", action="store_true",
                        help="是否启用文字方向分类（默认不启用，适合PPT场景）")

    args = parser.parse_args()

    ocr_processor = PPTOCRProcessor(lang="ch")

    if args.type == "image":
        print(f"开始处理图片：{args.input}")
        result = ocr_processor.process_single_image(args.input, use_cls=args.use_cls)
        print("识别结果：")
        for item in result:
            print(f"文字：{item['text']}，位置：{item['box']}，置信度：{item['confidence']}")
    elif args.type == "video":
        print(f"开始处理视频：{args.input}，间隔：{args.interval}秒")
        ocr_processor.process_video_frames(args.input, interval=args.interval, use_cls=args.use_cls)