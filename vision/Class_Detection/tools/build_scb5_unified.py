#!/usr/bin/env python3
"""Build a unified 3-class YOLO dataset from 9 SCB-5 subsets.

The 3-class system is designed for maximum robustness against the
incomplete-annotation problem inherent in SCB-5:
  0: student      (all students regardless of behavior)
  1: teacher      (all teacher activities)
  2: screen_board (blackboard + projection screen)

Behavioral state classification (active/focus/distracted) is handled
downstream by pose estimation + gaze + ST-GCN, NOT by the detector.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Global 3-Class Taxonomy
# ──────────────────────────────────────────────────────────────────
GLOBAL_NAMES = {
    0: "student",
    1: "teacher",
    2: "screen_board",
}

# Per-subset mapping: local_class_id → global_class_id
# Every human in student-oriented subsets → 0 (student)
# Every teacher label → 1 (teacher)
# Every board/screen label → 2 (screen_board)
SUBSET_MAPPINGS: dict[str, dict[int, int]] = {
    "SCB5-Handrise-Read-write-2024-9-17": {
        0: 0,   # hand_raising -> student
        1: 0,   # read -> student
        2: 0,   # write -> student
    },
    "SCB5-Stand-2024-9-17": {
        0: 0,   # stand -> student
    },
    "SCB5-Talk-2024-9-17": {
        0: 0,   # talk -> student
    },
    "SCB5-Discuss-2024-9-17": {
        0: 0,   # discuss -> student
    },
    "SCB5-Teacher-2024-9-17": {
        0: 1,   # teacher -> teacher
    },
    "SCB5-Teacher-Behavior-2024-9-17": {
        0: 1,   # guide -> teacher
        1: 0,   # answer -> student
        2: 0,   # stage_interact -> student
        3: 1,   # board_writing -> teacher
    },
    "SCB5-Talk-Teacher-Behavior-2024-9-17": {
        0: 0,   # talk -> student
        1: 1,   # guide -> teacher
        2: 0,   # answer -> student
        3: 0,   # stage_interact -> student
        4: 1,   # board_writing -> teacher
    },
    "SCB5-BlackBoard-Screen": {
        0: 2,   # blackboard -> screen_board
        1: 2,   # screen -> screen_board
    },
    "SCB5-BlackBoard-Sreen-Teacher": {
        0: 2,   # blackboard -> screen_board
        1: 2,   # screen -> screen_board
        2: 1,   # teacher -> teacher
    },
}

def sanitize_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    """Clip normalised YOLO box values into [0, 1] and reject degenerate boxes."""
    x2 = min(max(x, 0.0), 1.0)
    y2 = min(max(y, 0.0), 1.0)
    w2 = min(max(w, 0.0), 1.0)
    h2 = min(max(h, 0.0), 1.0)
    if w2 <= 0.0 or h2 <= 0.0:
        return None
    return x2, y2, w2, h2

def remap_label_lines(lines: list[str], mapping: dict[int, int]) -> tuple[list[str], int, int]:
    new_lines: list[str] = []
    clipped = 0
    dropped = 0

    for raw in lines:
        parts = raw.strip().split()
        if len(parts) < 5:
            dropped += 1
            continue
        try:
            local_cls = int(float(parts[0]))
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            dropped += 1
            continue

        global_cls = mapping.get(local_cls)
        if global_cls is None:
            dropped += 1
            continue

        sanitized = sanitize_box(x, y, w, h)
        if sanitized is None:
            dropped += 1
            continue

        sx, sy, sw, sh = sanitized
        if (sx, sy, sw, sh) != (x, y, w, h):
            clipped += 1

        new_lines.append(f"{global_cls} {sx:.6f} {sy:.6f} {sw:.6f} {sh:.6f}")

    return new_lines, clipped, dropped

def collect_split(src_root: Path, split: str) -> tuple[dict[str, Path], dict[str, list[str]]]:
    image_sources: dict[str, Path] = {}
    merged_labels: dict[str, list[str]] = defaultdict(list)

    stats = {
        "subsets_found": 0, "total_images": 0, "duplicates_merged": 0,
        "boxes_remapped": 0, "boxes_clipped": 0, "boxes_dropped": 0,
    }

    for subset_name, mapping in SUBSET_MAPPINGS.items():
        subset_dir = src_root / subset_name
        label_dir = None
        image_dir = None

        for candidate_label in [subset_dir / "labels" / split, subset_dir / "labels"]:
            if candidate_label.is_dir(): label_dir = candidate_label; break

        for candidate_image in [subset_dir / "images" / split, subset_dir / "images"]:
            if candidate_image.is_dir(): image_dir = candidate_image; break

        if label_dir is None or image_dir is None: continue

        stats["subsets_found"] += 1
        for label_path in sorted(label_dir.glob("*.txt")):
            stem = label_path.stem
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                if (image_dir / (stem + ext)).exists():
                    img_path = image_dir / (stem + ext)
                    break
            if not img_path: continue

            raw_lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            new_lines, clipped, dropped = remap_label_lines(raw_lines, mapping)

            stats["boxes_remapped"] += len(new_lines)
            stats["boxes_clipped"] += clipped
            stats["boxes_dropped"] += dropped

            if stem in image_sources: stats["duplicates_merged"] += 1
            else:
                image_sources[stem] = img_path
                stats["total_images"] += 1
            merged_labels[stem].extend(new_lines)

    print(f"  [{split}] subsets={stats['subsets_found']}, images={stats['total_images']}, "
          f"duplicates_merged={stats['duplicates_merged']}, boxes={stats['boxes_remapped']}")
    return image_sources, dict(merged_labels)

def write_split(image_sources: dict[str, Path], merged_labels: dict[str, list[str]], dst_root: Path, split: str, copy_images: bool) -> None:
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem, src_img in sorted(image_sources.items()):
        dst_img = dst_img_dir / src_img.name
        if not dst_img.exists():
            if copy_images: shutil.copy2(src_img, dst_img)
            else:
                try: dst_img.symlink_to(src_img.resolve())
                except OSError: shutil.copy2(src_img, dst_img)

        # Deduplicate identical boxes
        lines = merged_labels.get(stem, [])
        unique_lines = list(dict.fromkeys(lines))
        with (dst_lbl_dir / (stem + ".txt")).open("w", encoding="utf-8") as f:
            if unique_lines: f.write("\n".join(unique_lines) + "\n")

def print_class_distribution(dst_root: Path, split: str) -> None:
    lbl_dir = dst_root / "labels" / split
    if not lbl_dir.exists(): return
    counts: dict[int, int] = defaultdict(int)
    for txt in lbl_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").strip().splitlines():
            if line.strip(): counts[int(line.split()[0])] += 1
    print(f"\n  [{split}] Class distribution:")
    for cls_id in sorted(counts.keys()):
        print(f"    {cls_id:>2}: {GLOBAL_NAMES.get(cls_id, 'unknown'):<14s}  {counts[cls_id]:>6d} boxes")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[2] / "SCB-Dataset" / "SCB-Dataset")
    p.add_argument("--dst", type=Path, default=Path(__file__).resolve().parents[2] / "SCB5_yolo_unified")
    p.add_argument("--copy-images", action="store_true")
    args = p.parse_args()

    if not args.src.exists(): sys.exit(f"ERROR: {args.src} not found.")

    print(f"Building SCB-5 Unified 3-Class Dataset\n  Source : {args.src}\n  Target : {args.dst}\n")
    for split in ("train", "val"):
        print(f"Processing split: {split}")
        imgs, labels = collect_split(args.src, split)
        if imgs:
            write_split(imgs, labels, args.dst, split, args.copy_images)
            print_class_distribution(args.dst, split)
    print("\n✅ Dataset built! Use configs/scb_yolo.yaml for training.")

if __name__ == "__main__": main()
