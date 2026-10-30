from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SplitStats:
    split: str
    image_count: int
    label_count: int
    matched_pairs: int
    missing_label_files: int
    missing_image_files: int
    empty_label_files: int
    invalid_label_lines: int


@dataclass
class AuditReport:
    dataset_root: str
    classes_found: list[int]
    class_histogram: dict[int, int]
    split_stats: list[SplitStats]


def list_files(folder: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in set(suffixes)
    }


def parse_label_file(label_path: Path, class_counter: Counter[int]) -> tuple[int, int]:
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return 1, 0

    empty_file = 0
    invalid_lines = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            invalid_lines += 1
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            invalid_lines += 1
            continue

        # YOLO labels should be normalized to [0, 1].
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            invalid_lines += 1
            continue

        class_counter[cls] += 1

    return empty_file, invalid_lines


def audit_split(dataset_root: Path, split: str, class_counter: Counter[int]) -> SplitStats:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split folders for '{split}': {image_dir} or {label_dir}")

    images = list_files(image_dir, IMAGE_EXTENSIONS)
    labels = list_files(label_dir, {".txt"})

    image_stems = set(images)
    label_stems = set(labels)
    matched = image_stems & label_stems

    empty_label_files = 0
    invalid_label_lines = 0
    for stem in matched:
        empty_cnt, invalid_cnt = parse_label_file(labels[stem], class_counter)
        empty_label_files += empty_cnt
        invalid_label_lines += invalid_cnt

    return SplitStats(
        split=split,
        image_count=len(images),
        label_count=len(labels),
        matched_pairs=len(matched),
        missing_label_files=len(image_stems - label_stems),
        missing_image_files=len(label_stems - image_stems),
        empty_label_files=empty_label_files,
        invalid_label_lines=invalid_label_lines,
    )


def audit_dataset(dataset_root: Path) -> AuditReport:
    class_counter: Counter[int] = Counter()
    split_stats = [audit_split(dataset_root, "train", class_counter), audit_split(dataset_root, "val", class_counter)]

    return AuditReport(
        dataset_root=str(dataset_root),
        classes_found=sorted(class_counter.keys()),
        class_histogram=dict(sorted(class_counter.items())),
        split_stats=split_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO dataset integrity for SCB training.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("SCB_yolo_dataset"),
        help="Dataset root that contains images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional json output path.",
    )
    args = parser.parse_args()

    report = audit_dataset(args.dataset_root)

    json_payload = json.dumps(
        {
            "dataset_root": report.dataset_root,
            "classes_found": report.classes_found,
            "class_histogram": report.class_histogram,
            "split_stats": [asdict(x) for x in report.split_stats],
        },
        ensure_ascii=False,
        indent=2,
    )
    print(json_payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_payload, encoding="utf-8")


if __name__ == "__main__":
    main()

