import os
import shutil
from pathlib import Path

# Mapping from subset name to its local class indices -> target global class indices
# Based on configs/scb_yolo.yaml mapping:
# 0: hand_raising, 1: read, 2: write, 5: discuss, 6: talk, 7: answer, 8: stage_interact
# 13: stand, 15: teacher, 16: guide, 17: board_writing, 18: blackboard, 19: screen
SUBSET_MAPS = {
    "SCB5-Handrise-Read-write-2024-9-17": {0: 0, 1: 1, 2: 2},
    "SCB5-Stand-2024-9-17": {0: 13},
    "SCB5-Talk-2024-9-17": {0: 6},
    "SCB5-Talk-Teacher-Behavior-2024-9-17": {0: 6, 1: 16, 2: 7, 3: 8, 4: 17},
    "SCB5-Teacher-2024-9-17": {0: 15},
    "SCB5-Teacher-Behavior-2024-9-17": {0: 16, 1: 7, 2: 8, 3: 17},
    "SCB5-BlackBoard-Screen": {0: 18, 1: 19},
    "SCB5-BlackBoard-Sreen-Teacher": {0: 18, 1: 19, 2: 15},
    "SCB5-Discuss-2024-9-17": {0: 5}
}

def round_bbox(line):
    parts = line.strip().split()
    if len(parts) < 5: return line
    # class_id, x, y, w, h
    class_id = int(parts[0])
    # round to 4 decimal places to handle precision issues and avoid duplicated boxes
    bbox = [str(round(float(x), 4)) for x in parts[1:5]]
    return f"{class_id} " + " ".join(bbox)

def main():
    root_dir = Path("SCB-Dataset")
    target_dir = Path("SCB_yolo_dataset")
    
    if not root_dir.exists():
        print(f"Error: {root_dir} not found.")
        return

    # Create target directories
    for split in ["train", "val"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Starting dataset merge...")
    
    # Track annotations in memory to prevent identical duplicates 
    # Structure: labels_cache[split][filename] = set()
    labels_cache = {"train": {}, "val": {}}
    
    for subset, class_map in SUBSET_MAPS.items():
        subset_path = root_dir / subset
        if not subset_path.exists():
            print(f"Warning: Subset {subset} not found, skipping...")
            continue
            
        print(f"Processing subset: {subset}")
        
        for split in ["train", "val"]:
            img_dir = subset_path / "images" / split
            lbl_dir = subset_path / "labels" / split
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
                
            # Copy images
            for img_file in img_dir.glob("*.jpg"):
                target_img_path = target_dir / "images" / split / img_file.name
                if not target_img_path.exists():
                    shutil.copy2(img_file, target_img_path)
            
            # Merge labels
            for lbl_file in lbl_dir.glob("*.txt"):
                if lbl_file.name not in labels_cache[split]:
                    labels_cache[split][lbl_file.name] = set()
                    
                with open(lbl_file, "r") as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    orig_cls = int(parts[0])
                    if orig_cls in class_map:
                        target_cls = class_map[orig_cls]
                        new_line = f"{target_cls} {' '.join(parts[1:])}"
                        rounded = round_bbox(new_line)
                        labels_cache[split][lbl_file.name].add(rounded)

    print("Writing merged labels...")
    for split in ["train", "val"]:
        for filename, lines_set in labels_cache[split].items():
            out_file = target_dir / "labels" / split / filename
            # sort lines so we get a consistent output
            lines = sorted(list(lines_set), key=lambda x: int(x.split()[0]))
            with open(out_file, "w") as f:
                # restore the float precision by extracting from original parts or just using the rounded one
                # we'll write the rounded ones back since it's sufficient for YOLO and prevents bugs
                f.write("\n".join(lines) + "\n")

    print(f"Done! Merged dataset saved to {target_dir.absolute()}")

if __name__ == "__main__":
    main()
