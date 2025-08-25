#!/usr/bin/env python3
"""Create a small subset of DocLayNet (or similar YOLO-style dataset) for faster debugging.

Generates new folder structure:
  layout_data/<dataset_name>_mini/
    images/{train,val}
    labels/{train,val}
    train.txt
    val.txt

Selection strategy:
  - Sample first N lines from original train/val split lists (or random if --shuffle)
  - Optionally filter out label files with out-of-bounds (>1.0) normalized coords.

Usage:
  python create_mini_subset.py --source layout_data/doclaynet/doclaynet \
      --dataset-name doclaynet --train-n 1000 --val-n 200 --clean-labels
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import random


def parse_args():
    p = argparse.ArgumentParser(description="Create a mini subset for quick training")
    p.add_argument('--source', required=True, help='Path to original dataset root (contains images/, labels/, train.txt, val.txt)')
    p.add_argument('--dataset-name', default='doclaynet', help='Base dataset name')
    p.add_argument('--dest-root', default='layout_data', help='Destination parent directory')
    p.add_argument('--train-n', type=int, default=1000, help='Number of training images to sample')
    p.add_argument('--val-n', type=int, default=200, help='Number of validation images to sample')
    p.add_argument('--shuffle', action='store_true', help='Shuffle before sampling')
    p.add_argument('--clean-labels', action='store_true', help='Skip samples whose labels contain coords >1 or <0')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing subset')
    return p.parse_args()


def load_split_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")
    with path.open() as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def label_is_clean(label_file: Path) -> bool:
    if not label_file.exists():
        return False
    try:
        with label_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # class cx cy w h (normalized)
                nums = list(map(float, parts[1:5]))
                for v in nums:
                    if v < 0 or v > 1:
                        return False
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    src = Path(args.source)
    images_dir = src / 'images'
    labels_dir = src / 'labels'
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError('Source must contain images/ and labels/ directories')

    train_list = load_split_list(src / 'train.txt')
    val_list = load_split_list(src / 'val.txt')

    if args.shuffle:
        random.shuffle(train_list)
        random.shuffle(val_list)

    # Helper to map relative path forms: original lines may include absolute or relative
    def normalize_paths(lines: list[str]):
        norm = []
        for l in lines:
            p = Path(l)
            if p.is_absolute():
                # keep only filename
                norm.append(p.name)
            else:
                norm.append(p.name)
        return norm

    train_list = normalize_paths(train_list)
    val_list = normalize_paths(val_list)

    # Truncate
    train_sel = train_list[: args.train_n]
    val_sel = val_list[: args.val_n]

    dest = Path(args.dest_root) / f"{args.dataset_name}_mini"
    if dest.exists() and args.overwrite:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'images').mkdir(exist_ok=True)
    (dest / 'labels').mkdir(exist_ok=True)

    copied_train = []
    copied_val = []

    def copy_subset(subset: list[str], split_name: str, collector: list[str]):
        for fname in subset:
            stem = Path(fname).stem
            img_candidates = list(images_dir.glob(f"{stem}.*"))
            if not img_candidates:
                continue
            img_path = img_candidates[0]
            label_path = labels_dir / f"{stem}.txt"
            if args.clean_labels and not label_is_clean(label_path):
                continue
            # Copy
            dst_img = dest / 'images' / img_path.name
            dst_lbl = dest / 'labels' / label_path.name
            shutil.copy2(img_path, dst_img)
            if label_path.exists():
                shutil.copy2(label_path, dst_lbl)
            collector.append(img_path.name)

    copy_subset(train_sel, 'train', copied_train)
    copy_subset(val_sel, 'val', copied_val)

    # Write split files
    with (dest / 'train.txt').open('w') as f:
        for n in copied_train:
            f.write(str((dest / 'images' / n).resolve()) + '\n')
    with (dest / 'val.txt').open('w') as f:
        for n in copied_val:
            f.write(str((dest / 'images' / n).resolve()) + '\n')

    print(f"Mini subset created at: {dest}")
    print(f"Train images: {len(copied_train)}  Val images: {len(copied_val)}  (cleaned={args.clean_labels})")


if __name__ == '__main__':
    main()
