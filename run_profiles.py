#!/usr/bin/env python3
"""Unified training launcher with predefined profiles (light, full) for DocLayout-YOLO.

Use a single script and choose a profile instead of maintaining multiple runner files.

Profiles (tuned for ~4GB VRAM GPU like RTX 2050):
  light:
    - Fast iteration (~50 base epochs, optional short refinement)
    - Smaller model (yolov10n) and reduced image size for higher batch
  full:
    - Longer training (default 200 + 50 refinement)
    - Medium model (yolov10m) with moderate image size

You can override any parameter via CLI flags. Refinement stage is executed if refinement_epochs > 0.
"""
from run_experiment import ExperimentRunner
import argparse
import math


PROFILES = {
    "light": {
        "model": "n",
        "base_epochs": 50,
        "refinement_epochs": 10,
        "image_size": 512,
        "batch_size": 10,   # try 10; fallback 8 if OOM
        "lr0": 0.015,
        "mosaic": 0.3,
        "patience": 20,
        "mixed_precision": True,
        "workers": 3,
        "save_period": 25,
        "val_period": 5,
    },
    "full": {
        "model": "m",
        "base_epochs": 200,
        "refinement_epochs": 50,
        "image_size": 576,  # slightly lower than 640 to relieve VRAM
        "batch_size": 4,
        "lr0": 0.01,
        "mosaic": 0.6,
        "patience": 50,
        "mixed_precision": True,  # enable AMP to fit better
        "workers": 4,
        "save_period": 40,
        "val_period": 5,
    },
}


def build_arg_parser():
    p = argparse.ArgumentParser(description="DocLayout-YOLO profile runner (single file)")
    p.add_argument('--profile', choices=PROFILES.keys(), default='light', help='Chọn profile huấn luyện')
    p.add_argument('--data', default='doclaynet', help='Tên dataset (không cần .yaml nếu script nội bộ xử lý)')
    # Optional overrides
    p.add_argument('--model', default=None, help='Ghi đè kích thước model (n/s/m/l/x)')
    p.add_argument('--base-epochs', type=int, default=None, help='Số epoch base')
    p.add_argument('--refinement-epochs', type=int, default=None, help='Số epoch refinement (0 = bỏ qua)')
    p.add_argument('--imgsz', type=int, default=None, help='Image size (square)')
    p.add_argument('--batch', type=int, default=None, help='Batch size')
    p.add_argument('--lr0', type=float, default=None, help='Learning rate ban đầu')
    p.add_argument('--mosaic', type=float, default=None, help='Xác suất / hệ số mosaic')
    p.add_argument('--patience', type=int, default=None, help='Early stopping patience')
    p.add_argument('--amp', type=int, choices=[0, 1], default=None, help='Bật (1) / tắt (0) AMP')
    p.add_argument('--workers', type=int, default=None, help='Số dataloader workers')
    p.add_argument('--save-period', type=int, default=None, help='Chu kỳ lưu checkpoint')
    p.add_argument('--val-period', type=int, default=None, help='Chu kỳ validation')
    p.add_argument('--device', default='0', help='Thiết bị CUDA hoặc cpu')
    p.add_argument('--no-refine', action='store_true', help='Bỏ qua giai đoạn refinement (ưu tiên hơn flag refinement-epochs)')
    p.add_argument('--skip-dataset', action='store_true', help='Bỏ qua bước chuẩn bị dataset')
    return p


def merge_profile_args(args):
    prof = PROFILES[args.profile].copy()
    # Map CLI overrides
    mapping = [
        (args.model, 'model'),
        (args.base_epochs, 'base_epochs'),
        (args.refinement_epochs, 'refinement_epochs'),
        (args.imgsz, 'image_size'),
        (args.batch, 'batch_size'),
        (args.lr0, 'lr0'),
        (args.mosaic, 'mosaic'),
        (args.patience, 'patience'),
        (args.workers, 'workers'),
        (args.save_period, 'save_period'),
        (args.val_period, 'val_period'),
    ]
    for value, key in mapping:
        if value is not None:
            prof[key] = value
    if args.amp is not None:
        prof['mixed_precision'] = bool(args.amp)
    if args.no_refine:
        prof['refinement_epochs'] = 0
    # Derive save_period if not set explicitly for very short runs
    if 'save_period' not in prof or prof['save_period'] is None:
        prof['save_period'] = max(int(prof['base_epochs'] * 0.5), 10)
    return prof


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = merge_profile_args(args)

    runner = ExperimentRunner()

    overrides = {
        'batch_size': cfg['batch_size'],
        'image_size': cfg['image_size'],
        'base_epochs': cfg['base_epochs'],
        'refinement_epochs': cfg['refinement_epochs'],
        'lr0': cfg['lr0'],
        'mosaic': cfg['mosaic'],
        'patience': cfg['patience'],
        'mixed_precision': cfg['mixed_precision'],
        'save_period': cfg['save_period'],
        'val_period': cfg['val_period'],
        'workers': cfg['workers'],
        'device': args.device,
    }

    # Decide run mode
    if cfg['refinement_epochs'] > 0:
        runner.run_refinement_experiment('local_development', data=args.data, model=cfg['model'], **overrides)
    else:
        runner.run_baseline_experiment('local_development', data=args.data, model=cfg['model'], **overrides)


if __name__ == '__main__':
    main()
