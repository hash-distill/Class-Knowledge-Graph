#!/usr/bin/env python3
"""Merge and reduce dimensionality of SCB-Dataset labels.

Old classes (20):
  0: hand_raising, 1: read, 2: write, 3: bow_head, 4: turn_head,
  5: discuss, 6: talk, 7: answer, 8: stage_interact, 9: yawn,
  10: lean_desk, 11: use_phone, 12: use_computer, 13: stand, 14: clap,
  15: teacher, 16: guide, 17: board_writing, 18: blackboard, 19: screen

New classes (5):
  0: active_student (0, 5, 6, 7, 8, 14)
  1: focus_student (1, 2, 3, 12, 13)
  2: distracted_student (4, 9, 10, 11)
  3: teacher (15, 16, 17)
  4: screen_board (18, 19)
"""

import sys
import shutil
from pathlib import Path


CLASS_MAP = {
    0: 0, 5: 0, 6: 0, 7: 0, 8: 0, 14: 0,
    1: 1, 2: 1, 3: 1, 12: 1, 13: 1,
    4: 2, 9: 2, 10: 2, 11: 2,
    15: 3, 16: 3, 17: 3,
    18: 4, 19: 4
}


def main():
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "SCB_yolo_dataset"
    dst_dir = root / "SCB_yolo_dataset_merged"
    
    if not (src_dir / "labels").exists():
        print(f"Error: Label directory not found at {src_dir / 'labels'}")
        sys.exit(1)
        
    print(f"Starting dataset merge...")
    print(f"Source: O({src_dir})")
    print(f"Target: O({dst_dir})")
    
    # Create target dirs
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Process labels
    print("Mapping label classes...")
    for split in ["train", "val"]:
        src_split = src_dir / "labels" / split
        dst_split = dst_dir / "labels" / split
        
        if not src_split.exists():
            continue
            
        dst_split.mkdir(parents=True, exist_ok=True)
        txt_files = list(src_split.glob("*.txt"))
        
        for p in txt_files:
            new_lines = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_cls = int(parts[0])
                        new_cls = CLASS_MAP.get(old_cls, -1)
                        if new_cls != -1:
                            parts[0] = str(new_cls)
                            new_lines.append(" ".join(parts))
                            
            # Write mapped label
            with open(dst_split / p.name, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n" if new_lines else "")
        print(f"  [{split}] processed {len(txt_files)} label files.")

    # 2. Copy images (or handle logic if they don't want to duplicate)
    src_img = src_dir / "images"
    dst_img = dst_dir / "images"
    if src_img.exists() and not dst_img.exists():
        print("Copying images to merged dataset (this might take a moment)...")
        shutil.copytree(src_img, dst_img)
        print("Images copied successfully!")
    elif dst_img.exists():
        print("Images directory already exists in target. Skipping copy.")

    print(f"\nDone! Datatset ready at {dst_dir.name}")
    print("Please use '--data configs/scb_yolo_merged.yaml' in your next training run.")

if __name__ == "__main__":
    main()
