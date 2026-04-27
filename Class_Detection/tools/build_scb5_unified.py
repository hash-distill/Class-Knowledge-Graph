#!/usr/bin/env python3
"""Build a unified 13-class YOLO dataset from the 9 original SCB-5 subsets.

Each SCB-5 subset uses **local** class IDs that start from 0, so naively
merging label files produces class-ID collisions. This script remaps every
subset's local IDs to a single global 13-class taxonomy and deduplicates
images that appear in multiple subsets (merging their label lines).

Output layout (YOLO-compatible)::

    <dst_root>/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Usage::

    cd Class_Detection
    python tools/build_scb5_unified.py                       # defaults
    python tools/build_scb5_unified.py --src ../SCB-Dataset/SCB-Dataset --dst ../SCB5_yolo_unified
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Global 13-class taxonomy
# ──────────────────────────────────────────────────────────────────
#   0: hand_raising      6: stage_interact   11: blackboard
#   1: read              7: stand            12: screen
#   2: write             8: teacher
#   3: discuss           9: guide
#   4: talk             10: board_writing
#   5: answer
#
# Roles:
#   Students   : 0-7
#   Teachers   : 8-10
#   Environment: 11-12
# ──────────────────────────────────────────────────────────────────

# Per-subset mapping: local_class_id → global_class_id
SUBSET_MAPPINGS: dict[str, dict[int, int]] = {
    "SCB5-Handrise-Read-write-2024-9-17": {
        0: 0,   # hand_raising
        1: 1,   # read
        2: 2,   # write
    },
    "SCB5-Stand-2024-9-17": {
        0: 7,   # stand
    },
    "SCB5-Talk-2024-9-17": {
        0: 4,   # talk
    },
    "SCB5-Discuss-2024-9-17": {
        0: 3,   # discuss
    },
    "SCB5-Teacher-2024-9-17": {
        0: 8,   # teacher
    },
    "SCB5-Teacher-Behavior-2024-9-17": {
        0: 9,   # guide
        1: 5,   # answer
        2: 6,   # stage_interact
        3: 10,  # board_writing
    },
    "SCB5-Talk-Teacher-Behavior-2024-9-17": {
        0: 4,   # talk
        1: 9,   # guide
        2: 5,   # answer
        3: 6,   # stage_interact
        4: 10,  # board_writing
    },
    "SCB5-BlackBoard-Screen": {
        0: 11,  # blackboard
        1: 12,  # screen
    },
    "SCB5-BlackBoard-Sreen-Teacher": {
        0: 11,  # blackboard
        1: 12,  # screen
        2: 8,   # teacher
    },
}

GLOBAL_NAMES = {
    0: "hand_raising",
    1: "read",
    2: "write",
    3: "discuss",
    4: "talk",
    5: "answer",
    6: "stage_interact",
    7: "stand",
    8: "teacher",
    9: "guide",
    10: "board_writing",
    11: "blackboard",
    12: "screen",
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def sanitize_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    """Clip normalised YOLO box values into [0, 1] and reject degenerate boxes."""
    x2 = min(max(x, 0.0), 1.0)
    y2 = min(max(y, 0.0), 1.0)
    w2 = min(max(w, 0.0), 1.0)
    h2 = min(max(h, 0.0), 1.0)
    if w2 <= 0.0 or h2 <= 0.0:
        return None
    return x2, y2, w2, h2


def remap_label_lines(
    lines: list[str],
    mapping: dict[int, int],
) -> tuple[list[str], int, int]:
    """Remap one label file's lines. Returns (new_lines, clipped_count, dropped_count)."""
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


# ──────────────────────────────────────────────────────────────────
# Core builder
# ──────────────────────────────────────────────────────────────────

def collect_split(
    src_root: Path,
    split: str,
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Scan all subsets for a given split and return merged image→path and image→label-lines.

    Deduplication: if the same image filename appears in multiple subsets we
    keep the first copy of the image file and **merge** the remapped label
    lines from all subsets.
    """
    image_sources: dict[str, Path] = {}          # stem → first image path
    merged_labels: dict[str, list[str]] = defaultdict(list)  # stem → label lines

    stats = {
        "subsets_found": 0,
        "total_images": 0,
        "duplicates_merged": 0,
        "boxes_remapped": 0,
        "boxes_clipped": 0,
        "boxes_dropped": 0,
    }

    for subset_name, mapping in SUBSET_MAPPINGS.items():
        subset_dir = src_root / subset_name

        # Each subset may organise splits differently
        # Try: labels/{split}/, labels/ (flat), images/{split}/, images/ (flat)
        label_dir = None
        image_dir = None

        for candidate_label in [
            subset_dir / "labels" / split,
            subset_dir / "labels",
        ]:
            if candidate_label.is_dir():
                label_dir = candidate_label
                break

        for candidate_image in [
            subset_dir / "images" / split,
            subset_dir / "images",
        ]:
            if candidate_image.is_dir():
                image_dir = candidate_image
                break

        if label_dir is None or image_dir is None:
            continue

        stats["subsets_found"] += 1
        txt_files = sorted(label_dir.glob("*.txt"))

        for label_path in txt_files:
            stem = label_path.stem

            # Find matching image (try common extensions)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                candidate = image_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                # Label without image — skip
                continue

            # Read and remap labels
            raw_lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            new_lines, clipped, dropped = remap_label_lines(raw_lines, mapping)

            stats["boxes_remapped"] += len(new_lines)
            stats["boxes_clipped"] += clipped
            stats["boxes_dropped"] += dropped

            if stem in image_sources:
                stats["duplicates_merged"] += 1
            else:
                image_sources[stem] = img_path
                stats["total_images"] += 1

            merged_labels[stem].extend(new_lines)

    print(f"  [{split}] subsets={stats['subsets_found']}, "
          f"images={stats['total_images']}, "
          f"duplicates_merged={stats['duplicates_merged']}, "
          f"boxes={stats['boxes_remapped']}, "
          f"clipped={stats['boxes_clipped']}, "
          f"dropped={stats['boxes_dropped']}")

    return image_sources, dict(merged_labels)


def write_split(
    image_sources: dict[str, Path],
    merged_labels: dict[str, list[str]],
    dst_root: Path,
    split: str,
    copy_images: bool,
) -> None:
    """Write the merged dataset for one split."""
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem, src_img in sorted(image_sources.items()):
        # Copy / link image
        dst_img = dst_img_dir / src_img.name
        if not dst_img.exists():
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                try:
                    dst_img.symlink_to(src_img.resolve())
                except OSError:
                    shutil.copy2(src_img, dst_img)

        # Write merged & deduplicated labels
        lines = merged_labels.get(stem, [])
        # Deduplicate identical label lines (same class + same box)
        unique_lines = list(dict.fromkeys(lines))
        label_path = dst_lbl_dir / (stem + ".txt")
        with label_path.open("w", encoding="utf-8") as f:
            if unique_lines:
                f.write("\n".join(unique_lines) + "\n")


def print_class_distribution(dst_root: Path, split: str) -> None:
    """Print class distribution for verification."""
    lbl_dir = dst_root / "labels" / split
    if not lbl_dir.exists():
        return

    counts: dict[int, int] = defaultdict(int)
    for txt in lbl_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1
                except ValueError:
                    pass

    print(f"\n  [{split}] Class distribution:")
    for cls_id in sorted(counts.keys()):
        name = GLOBAL_NAMES.get(cls_id, f"unknown_{cls_id}")
        print(f"    {cls_id:>2}: {name:<16s}  {counts[cls_id]:>6d} boxes")
    print(f"    {'':>2}  {'TOTAL':<16s}  {sum(counts.values()):>6d} boxes")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build unified 13-class YOLO dataset from SCB-5 subsets."
    )
    p.add_argument(
        "--src", type=Path,
        default=Path(__file__).resolve().parents[2] / "SCB-Dataset" / "SCB-Dataset",
        help="Root directory containing the 9 SCB-5 subset folders.",
    )
    p.add_argument(
        "--dst", type=Path,
        default=Path(__file__).resolve().parents[2] / "SCB5_yolo_unified",
        help="Output directory for the unified dataset.",
    )
    p.add_argument(
        "--copy-images", action="store_true", default=False,
        help="Copy image files instead of symlinking (slower but more portable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src
    dst_root = args.dst

    if not src_root.exists():
        print(f"ERROR: Source directory not found: {src_root}")
        sys.exit(1)

    print("=" * 60)
    print("  Building SCB-5 Unified 13-Class Dataset")
    print("=" * 60)
    print(f"  Source : {src_root}")
    print(f"  Target : {dst_root}")
    print(f"  Classes: {len(GLOBAL_NAMES)}")
    print()

    for split in ("train", "val"):
        print(f"Processing split: {split}")
        image_sources, merged_labels = collect_split(src_root, split)

        if not image_sources:
            print(f"  WARNING: No images found for split '{split}'. Skipping.")
            continue

        write_split(image_sources, merged_labels, dst_root, split, args.copy_images)
        print_class_distribution(dst_root, split)
        print()

    print("=" * 60)
    print("  ✅ Dataset built successfully!")
    print(f"  Use configs/scb_yolo.yaml for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
