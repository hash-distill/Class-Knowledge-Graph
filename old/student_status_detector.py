# student_status_detector.py
import os
import cv2
import time
from ultralytics import YOLO
from typing import List, Dict


class StudentStatusDetector:
    def __init__(self, model_path: str = "yolov11n.pt"):
        """初始化YOLO11模型，支持自定义训练模型"""
        # 若使用自定义训练的学生状态模型，替换model_path为本地权重文件（如"yolov11_student_best.pt"）
        self.model = YOLO(model_path)
        # 状态类别（需与训练时的标签一致，示例：0=举手，1=专注，2=走神，3=低头）
        self.class_names = ["hand_up", "focus", "distracted", "bow_down"]
        print("YOLO11模型加载完成，支持类别:", self.class_names)

    def detect_single_frame(self, frame: cv2.Mat, conf_threshold: float = 0.5) -> List[Dict]:
        """处理单帧图像，返回检测结果"""
        # 模型推理（返回目标框、类别、置信度）
        results = self.model(frame, conf=conf_threshold)

        # 解析结果
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 目标框坐标
                cls_id = int(box.cls[0])  # 类别ID
                conf = float(box.conf[0])  # 置信度

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class": self.class_names[cls_id],
                    "confidence": round(conf, 3)
                })
        return detections

    def process_video(self, video_path: str, output_path: str = None) -> List[Dict]:
        """处理视频文件，输出每帧的检测结果，可选保存带标注的视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 若需保存输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测学生状态
            timestamp = frame_count / fps
            detections = self.detect_single_frame(frame)

            # 保存结果
            results.append({
                "timestamp": round(timestamp, 2),
                "frame_idx": frame_count,
                "detections": detections
            })

            # 标注并保存视频（可选）
            if out:
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    cls = det["class"]
                    conf = det["confidence"]
                    # 画框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls} {conf}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out.write(frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧")

        cap.release()
        if out:
            out.release()
            print(f"标注视频已保存至: {output_path}")

        return results


# 测试代码
if __name__ == "__main__":
    # 若使用自定义模型，替换为训练好的权重路径
    detector = StudentStatusDetector(model_path="yolov11_student_best.pt")

    # 测试单帧图片
    frame = cv2.imread("classroom_frame.jpg")
    det_result = detector.detect_single_frame(frame)
    print("单帧检测结果:", det_result)

    # 测试视频处理（输出标注视频）
    video_results = detector.process_video(
        video_path="classroom_lecture.mp4",
        output_path="classroom_annotated.mp4"
    )
    print(f"视频处理完成，共 {len(video_results)} 帧结果")