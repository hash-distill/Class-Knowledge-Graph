import cv2
import argparse
import json
import time
import numpy as np
from pathlib import Path
from loguru import logger
from analyzer import SAGEAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="SAGE-Net学生状态识别系统")
    parser.add_argument("--camera_id", default=0, type=int,
                        help="摄像头ID或视频文件路径")
    parser.add_argument("--config", default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--model_path", default="models/sage_net.pth",
                        help="SAGE-Net模型权重路径")
    parser.add_argument("--display", action="store_true",
                        help="显示实时画面")
    parser.add_argument("--output", default="outputs/engagement_log.jsonl",
                        help="输出文件路径")
    parser.add_argument("--fps", type=int, default=30,
                        help="处理帧率")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置"""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("student_engagement", {})
    except Exception as e:
        logger.warning(f"配置加载失败: {e}，使用默认配置")
        return {}


def setup_output_dir(output_path: str) -> str:
    """设置输出目录"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_path


def draw_results(frame: np.ndarray, results: dict) -> np.ndarray:
    """
    在帧上绘制SAGE-Net分析结果

    Args:
        frame: 输入帧
        results: 学生状态字典

    Returns:
        绘制后的帧
    """
    overlay = frame.copy()

    for student in results.values():
        bbox = student.get("bbox", [50, 50, 150, 150])
        x1, y1, x2, y2 = bbox

        # 根据参与度选择颜色
        score = student["engagement_score"]
        if score >= 80:
            color = (0, 255, 0)  # 绿色 - 高度专注
        elif score >= 60:
            color = (0, 255, 255)  # 黄色 - 中度参与
        else:
            color = (0, 0, 255)  # 红色 - 低参与度

        # 绘制边界框
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 绘制信息背景
        info_lines = [
            f"ID:{student['student_id']} | Score:{score:.1f}",
            f"{student['emotion']} | {student['attention']}",
            f"Gaze:({student['gaze'][0]:.2f},{student['gaze'][1]:.2f})"
        ]

        text_y = y1 - 10
        for line in info_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 背景
            cv2.rectangle(overlay,
                          (x1, text_y - text_h - 5),
                          (x1 + text_w + 10, text_y + 5),
                          (0, 0, 0), -1)

            # 文字
            cv2.putText(overlay, line, (x1 + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_y -= text_h + 5

    # 添加半透明效果
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    return frame


def draw_stats(frame: np.ndarray, stats: dict):
    """绘制统计信息"""
    h, w = frame.shape[:2]

    # 背景条
    cv2.rectangle(frame, (0, h - 100), (400, h), (0, 0, 0), -1)

    lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Active Students: {stats.get('active_students', 0)}",
        f"Avg Engagement: {stats.get('avg_engagement', 0):.1f}%",
        f"Inference: {stats.get('avg_inference_time', 0):.1f}ms"
    ]

    y = h - 80
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 20

    return frame


def capture_frame(source):
    """帧捕获生成器"""
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"无法打开视频源: {source}")
        return

    # 设置参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yield frame

    cap.release()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)
    config["model_path"] = args.model_path

    # 初始化输出
    output_file = setup_output_dir(args.output)

    # 初始化SAGEAnalyzer
    analyzer = SAGEAnalyzer(args.model_path, config=config)

    # 帧率控制
    frame_interval = 1.0 / args.fps
    last_time = time.time()
    frame_count = 0

    # 性能监控
    fps_counter = 0
    fps_start = time.time()

    try:
        for frame in capture_frame(args.camera_id):
            current_time = time.time()

            # 帧率控制
            if current_time - last_time < frame_interval:
                continue

            last_time = current_time
            frame_count += 1

            # FPS计算
            fps_counter += 1
            if fps_counter >= 30:  # 每30帧计算一次FPS
                fps = fps_counter / (current_time - fps_start)
                fps_counter = 0
                fps_start = current_time
            else:
                fps = 0

            # 缩放帧以提升速度
            process_frame = cv2.resize(frame, (640, 360))

            # SAGE-Net分析
            results = analyzer.analyze_frame(process_frame)

            # 绘制结果
            display_frame = frame.copy()
            if args.display and results:
                display_frame = draw_results(display_frame, results)

                # 绘制统计
                stats = analyzer.get_stats()
                stats["fps"] = fps
                display_frame = draw_stats(display_frame, stats)

                cv2.imshow("SAGE-Net Student Analysis", display_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            # 保存结果
            if results:
                log_entry = {
                    "timestamp": time.time(),
                    "fps": fps,
                    "students": list(results.values())
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    json.dump(log_entry, f, ensure_ascii=False)
                    f.write("\n")

            

    except KeyboardInterrupt:
        
    except Exception as e:
        logger.error(f"主循环异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        final_stats = analyzer.get_stats()

        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()