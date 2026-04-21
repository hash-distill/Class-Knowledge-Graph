"""Train ST-GCN for classroom action recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stgcn import STGCN
from models.graph import Graph


class KeypointDataset(Dataset):
    """Load pre-extracted keypoint sequences (.npy files).

    Expected directory structure::

        root/
        ├── train/
        │   ├── writing_001.npy      # shape (C=3, T=30, V=17, M=1)
        │   ├── hand_raising_002.npy
        │   └── ...
        └── label.json               # {"writing_001": 0, ...}
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        self.root = root / split
        label_file = root / "label.json"

        if label_file.exists():
            self.labels = json.loads(label_file.read_text(encoding="utf-8"))
        else:
            self.labels = {}

        self.samples: list[tuple[Path, int]] = []
        if self.root.exists():
            for npy in sorted(self.root.glob("*.npy")):
                stem = npy.stem
                label = self.labels.get(stem, 0)
                self.samples.append((npy, int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        data = np.load(str(path)).astype(np.float32)  # (C, T, V, M)
        return torch.from_numpy(data), label


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ST-GCN for classroom action recognition.")
    p.add_argument("--keypoints-dir", type=Path, required=True,
                   help="Root dir with train/ val/ and label.json")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to YAML config (overrides defaults)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument("--save-dir", type=Path, default=Path("artifacts/runs/stgcn"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load config if provided
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if "training" in cfg:
                args.epochs = cfg["training"].get("epochs", args.epochs)
                args.batch = cfg["training"].get("batch_size", args.batch)
                args.lr = cfg["training"].get("learning_rate", args.lr)
            if "model" in cfg:
                args.num_classes = cfg["model"].get("num_classes", args.num_classes)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    graph = Graph(layout="coco", strategy="spatial")
    model = STGCN(in_channels=3, num_classes=args.num_classes, graph=graph,
                  edge_importance=True, dropout=0.3).to(device)

    train_ds = KeypointDataset(args.keypoints_dir, "train")
    val_ds = KeypointDataset(args.keypoints_dir, "val")

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Check --keypoints-dir structure.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4) if len(val_ds) > 0 else None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_acc = eval_epoch(model, val_loader, device) if val_loader else 0.0

        print(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_dir / "best.pt")

    torch.save(model.state_dict(), args.save_dir / "last.pt")
    print(f"Training complete. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
