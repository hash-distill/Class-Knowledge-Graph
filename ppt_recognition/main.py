import cv2
import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger
from processor import PPTProcessor, SlideInfo


def parse_args():
    parser = argparse.ArgumentParser(description="PPT内容实时识别系统")
    parser.add_argument("--source", default=0,
                        help="视频源: 0为摄像头, 'screen'为屏幕, 或视频文件路径")
    parser.add_argument("--config", default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--output", default="outputs/knowledge_points.jsonl",
                        help="输出文件路径")
    parser.add_argument("--display", action="store_true",
                        help="显示实时画面")
    parser.add_argument("--fps", type=int, default=30,
                        help="处理帧率限制")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("ppt_recognition", {})
    except Exception as e:
        logger.warning(f"无法加载配置文件 {config_path}: {e}，使用默认配置")
        return {}


def setup_output_dir(output_path: str):
    """设置输出目录"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_path


def capture_frame(source):
    """捕获帧的生成器"""
    if source == "screen":
        # 屏幕捕获
        try:
            import mss
            sct = mss.mss()
            monitor = sct.monitors[1]  # 主屏幕

            while True:
                img = sct.grab(monitor)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                yield frame
        except ImportError:
            logger.error("未安装mss库，无法使用屏幕捕获")
            return
    else:
        # 视频/摄像头捕获
        try:
            source = int(source)  # 尝试转换为摄像头ID
        except ValueError:
            pass  # 保持为文件路径

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            return

        # 设置缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            yield frame

        cap.release()


def draw_result(frame: np.ndarray, slide_info: SlideInfo):
    """在帧上绘制识别结果"""
    if slide_info is None:
        return frame

    # 绘制文本背景
    text = f"知识点: {slide_info.knowledge_point}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(frame, (10, 10), (10 + text_w + 10, 10 + text_h + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 15 + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 初始化输出目录
    output_file = setup_output_dir(args.output)

    # 初始化处理器
    processor = PPTProcessor(gpu_id=0, config=config)

    # 帧率控制
    frame_interval = 1.0 / args.fps
    last_time = time.time()
    frame_count = 0

    # 主循环
    try:
        for frame in capture_frame(args.source):
            current_time = time.time()

            # 帧率控制
            if current_time - last_time < frame_interval:
                continue

            last_time = current_time
            frame_count += 1

            # 处理帧
            if frame_count % 3 == 0:  # 每3帧处理一次
                slide_info = processor.process_frame(frame)

                if slide_info:
                    # 保存结果
                    with open(output_file, "a", encoding="utf-8") as f:
                        json.dump(slide_info.__dict__, f, ensure_ascii=False)
                        f.write("\n")

                    

                # 显示
                if args.display:
                    display_frame = draw_result(frame.copy(), slide_info)
                    cv2.imshow("PPT Recognition", display_frame)

                    if cv2.waitKey(1) == ord('q'):
                        break

        

    except KeyboardInterrupt:
        
    except Exception as e:
        logger.error(f"主循环异常: {e}")
    finally:
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()