"""Train YOLO26 detector on SCB-Dataset5 (or compatible YOLO dataset)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def train_detector(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        patience=args.patience,
        cache=args.cache,
        pretrained=args.pretrained,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26 detector on SCB-Dataset5.")
    p.add_argument("--data", type=Path, default=Path("configs/scb_yolo.yaml"))
    p.add_argument("--model", type=str, default="yolo26s.pt", help="YOLO26 base model.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=Path, default=Path("artifacts/runs/detect"))
    p.add_argument("--name", type=str, default="scb_yolo26s")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_detector(args)


if __name__ == "__main__":
    main()
