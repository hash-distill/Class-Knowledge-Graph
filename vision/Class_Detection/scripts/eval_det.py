"""Evaluate trained YOLO26 model on a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO26 detector.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("configs/scb_yolo.yaml"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
