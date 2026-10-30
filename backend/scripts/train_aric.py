"""Train a bounded ARIC frame-presence baseline; requires only Python and NumPy."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning.aric import prepare_dataset, train_from_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\Research\Class Detection Dataset"),
                        help="Directory containing ARIC_supplement, or ARIC_supplement itself")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/learning/aric")
    parser.add_argument("--max-images", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-support", type=int, default=15, help="Minimum TRAIN positive AND negative frames per label")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--prepare-only", action="store_true", help="Only audit/stream/cache a portable training bundle")
    parser.add_argument("--from-bundle", type=Path, help="Train from cached NPZ without original archive or annotations")
    args = parser.parse_args()
    bundle = args.from_bundle or prepare_dataset(args.dataset_root, args.output, args.max_images, args.seed)
    if not args.prepare_only:
        train_from_bundle(bundle, args.output, min_support=args.min_support, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
