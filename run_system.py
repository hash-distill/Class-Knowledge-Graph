#!/usr/bin/env python3
"""
课堂视觉识别系统 - 一键启动脚本
同时启动PPT识别和SAGE-Net学生状态识别
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
import signal
from loguru import logger


class ClassroomVisionSystem:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.processes = []

        # 检查依赖

        self._check_dependencies()
        self._prepare_outputs()

    def _check_dependencies(self):
        """检查必要依赖"""
        required_files = [
            "ppt_recognition/main.py",
            "student_engagement/main.py",
            "config.yaml"
        ]

        for file in required_files:
            if not Path(file).exists():
                logger.error(f"缺少必要文件: {file}")
                sys.exit(1)


    def _prepare_outputs(self):
        """准备输出目录"""
        output_dirs = ["outputs", "logs", "models"]
        for dir_name in output_dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def start_ppt_recognition(self):
        """启动PPT识别服务"""
        cmd = [
            sys.executable, "ppt_recognition/main.py",
            "--source", "screen",
            "--config", self.config_file,
            "--output", "outputs/knowledge_points.jsonl",
            "--display"
        ]

        proc = subprocess.Popen(cmd, stdout="logs/ppt.log", stderr="logs/ppt.log")
        self.processes.append(("PPT", proc))

    def start_sage_net(self):
        """启动SAGE-Net服务"""

        # 检查模型文件
        model_path = "models/sage_net.pth"
        if not Path(model_path).exists():
            logger.warning(f"SAGE-Net模型文件不存在: {model_path}")
            logger.warning("请使用 train_sage.py 训练模型或使用预训练模型")

        cmd = [
            sys.executable, "student_engagement/main.py",
            "--camera_id", "0",
            "--config", self.config_file,
            "--model_path", model_path,
            "--output", "outputs/engagement_log.jsonl",
            "--display"
        ]

        proc = subprocess.Popen(cmd, stdout="logs/sage.log", stderr="logs/sage.log")
        self.processes.append(("SAGE", proc))

    def start_monitor(self):
        """启动监控进程"""

        try:
            while True:
                # 检查进程状态
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        logger.error(f"{name}进程异常退出 (code: {proc.poll()})")
                        self.stop()
                        return

                # 每秒检查一次
                time.sleep(1)

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """停止所有进程"""

        for name, proc in self.processes:
            if proc.poll() is None:
                
                proc.terminate()
                proc.wait()


    def start(self, modules: list = None):
        """
        启动系统

        Args:
            modules: 要启动的模块列表 ["ppt", "sage"]，None表示全部
        """
        if modules is None:
            modules = ["ppt", "sage"]


        # 启动指定模块
        if "ppt" in modules:
            self.start_ppt_recognition()
            time.sleep(2)  # 等待初始化

        if "sage" in modules:
            self.start_sage_net()
            time.sleep(2)

        # 启动监控
        self.start_monitor()


def main():
    parser = argparse.ArgumentParser(description="课堂视觉识别系统启动器")
    parser.add_argument("--modules", nargs="+", choices=["ppt", "sage"],
                        help="要启动的模块")
    parser.add_argument("--config", default="config.yaml",
                        help="配置文件路径")

    args = parser.parse_args()

    system = ClassroomVisionSystem(args.config)

    # 注册信号处理
    def signal_handler(sig, frame):
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动系统
    system.start(args.modules)


if __name__ == "__main__":
    main()