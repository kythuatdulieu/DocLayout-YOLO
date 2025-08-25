#!/usr/bin/env python3
"""Quick (light) training pipeline for DocLayout-YOLO on low-VRAM GPU (~4GB).

Goals:
- Fast iteration (≈ 50 base epochs, optional small refinement stage)
- Fit comfortably in 4GB (RTX 2050)
- Use mixed precision to maximize effective batch size

This script wraps the existing ExperimentRunner with overrides tuned for speed.
"""
from run_experiment import ExperimentRunner
import argparse


def main():
    parser = argparse.ArgumentParser(description="Light (fast) DocLayout-YOLO run")
    parser.add_argument('--data', default='doclaynet', help='Dataset name (expects corresponding .yaml)')
    parser.add_argument('--model', default='n', help='Model size (n/s/m) – default n for speed')
    parser.add_argument('--epochs', type=int, default=50, help='Total base epochs')
    parser.add_argument('--refine-epochs', type=int, default=10, help='Refinement epochs (0 to skip refinement stage)')
    parser.add_argument('--imgsz', type=int, default=512, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (adjust if OOM)')
    parser.add_argument('--device', default='0', help='CUDA device id or cpu')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--mosaic', type=float, default=0.5, help='Mosaic augmentation prob')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial LR')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--amp', type=int, default=1, help='Use AMP (1) or not (0)')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset setup step')
    args = parser.parse_args()

    runner = ExperimentRunner()

    overrides = {
        'batch_size': args.batch,
        'image_size': args.imgsz,
        'base_epochs': args.epochs,
        'refinement_epochs': args.refine_epochs,
        'lr0': args.lr0,
        'mosaic': args.mosaic,
        'patience': args.patience,
        'mixed_precision': bool(args.amp),
        'save_period': max(args.epochs // 2, 10),
        'val_period': 5,
        'workers': args.workers,
        'device': args.device,
    }

    # Run combined (baseline+refinement) or only baseline if refinement_epochs == 0
    if args.refine_epochs > 0:
        runner.run_refinement_experiment('local_development', data=args.data, model=args.model, **overrides)
    else:
        runner.run_baseline_experiment('local_development', data=args.data, model=args.model, **overrides)


if __name__ == '__main__':
    main()
