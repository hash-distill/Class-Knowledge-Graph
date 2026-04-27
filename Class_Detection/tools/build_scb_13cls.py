#!/usr/bin/env python3
"""Build a 13-class SCB YOLO dataset from the original SCB labels.

The original SCB dataset contains 20 class IDs, but this workspace currently
has labels for only 13 IDs. This script remaps those 13 IDs to contiguous IDs
0..12 and writes a clean dataset for training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Original IDs present in this dataset -> new contiguous IDs.
# new_id: class_name (old_id)
# 0: hand_raising (0)
# 1: read (1)
# 2: write (2)
# 3: discuss (5)
# 4: talk (6)
# 5: answer (7)
# 6: stage_interact (8)
# 7: stand (13)
# 8: teacher (15)
# 9: guide (16)
# 10: board_writing (17)
# 11: blackboard (18)
# 12: screen (19)
CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    13: 7,
    15: 8,
    16: 9,
    17: 10,
    18: 11,
    19: 12,
}


def sanitize_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    """Clip normalized YOLO box values into [0, 1] and reject degenerate boxes."""
    x2 = min(max(x, 0.0), 1.0)
    y2 = min(max(y, 0.0), 1.0)
    w2 = min(max(w, 0.0), 1.0)
    h2 = min(max(h, 0.0), 1.0)
    if w2 <= 0.0 or h2 <= 0.0:
        return None
    return x2, y2, w2, h2


def remap_split(src_split: Path, dst_split: Path) -> tuple[int, int, int, int]:
    """Remap one split and return (num_files, num_boxes, num_clipped, num_dropped)."""
    dst_split.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(src_split.glob("*.txt"))
    box_count = 0
    clipped_count = 0
    dropped_count = 0

    for label_path in txt_files:
        new_lines: list[str] = []
        with label_path.open("r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) < 5:
                    dropped_count += 1
                    continue
                try:
                    old_cls = int(float(parts[0]))
                    x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                except ValueError:
                    dropped_count += 1
                    continue

                new_cls = CLASS_MAP.get(old_cls)
                if new_cls is None:
                    dropped_count += 1
                    continue

                sanitized = sanitize_box(x, y, w, h)
                if sanitized is None:
                    dropped_count += 1
                    continue

                sx, sy, sw, sh = sanitized
                if (sx, sy, sw, sh) != (x, y, w, h):
                    clipped_count += 1

                new_lines.append(f"{new_cls} {sx:.6f} {sy:.6f} {sw:.6f} {sh:.6f}")

        with (dst_split / label_path.name).open("w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

        box_count += len(new_lines)

    return len(txt_files), box_count, clipped_count, dropped_count


import shutil

def ensure_images_link(src_images: Path, dst_images: Path) -> None:
    """Create a symlink to source images (fallback to copying if unsupported)."""
    if dst_images.is_dir():
        return
    # On Windows, Git sometimes checks out symlinks as plain text files containing the path
    if dst_images.exists() and not dst_images.is_dir():
        print(f"Warning: {dst_images} is a file (likely a broken git symlink). Deleting it...")
        dst_images.unlink()
        
    try:
        dst_images.symlink_to(src_images, target_is_directory=True)
        print("Images linked via symlink.")
    except OSError as e:
        print(f"Symlink failed ({e}). Falling back to copying images. This may take a while...")
        shutil.copytree(src_images, dst_images)
        print("Images copied successfully.")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "SCB_yolo_dataset"
    dst_dir = root / "SCB_yolo_dataset_13cls"

    if not (src_dir / "labels").exists() or not (src_dir / "images").exists():
        print(f"Error: source dataset not found: {src_dir}")
        sys.exit(1)

    print("Building SCB 13-class dataset...")
    print(f"Source: {src_dir}")
    print(f"Target: {dst_dir}")

    for split in ("train", "val"):
        src_split = src_dir / "labels" / split
        if not src_split.exists():
            print(f"Skip split: {split} (missing source labels)")
            continue
        dst_split = dst_dir / "labels" / split
        num_files, num_boxes, num_clipped, num_dropped = remap_split(src_split, dst_split)
        print(
            f"[{split}] files={num_files}, remapped_boxes={num_boxes}, "
            f"clipped_boxes={num_clipped}, dropped_rows={num_dropped}"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    ensure_images_link(src_dir / "images", dst_dir / "images")
    print("Done. Use configs/scb_yolo_13cls.yaml for training.")


if __name__ == "__main__":
    main()
