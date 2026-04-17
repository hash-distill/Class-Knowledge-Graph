"""Fine-tune YOLO26-Pose on classroom-specific pose data (optional)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO26-Pose model.")
    p.add_argument("--data", type=Path, required=True, help="Pose dataset YAML.")
    p.add_argument("--model", type=str, default="yolo26n-pose.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=Path, default=Path("artifacts/runs/pose"))
    p.add_argument("--name", type=str, default="classroom_pose")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
