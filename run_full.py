#!/usr/bin/env python3
"""Full (long) training pipeline wrapper.

Designed when you want a more thorough training (larger model + more epochs) but still
bounded by ~4GB VRAM. Adjusts batch size automatically if OOM is detected (future todo).
"""
from run_experiment import ExperimentRunner
import argparse


def main():
    parser = argparse.ArgumentParser(description="Full DocLayout-YOLO training wrapper")
    parser.add_argument('--data', default='doclaynet', help='Dataset name')
    parser.add_argument('--model', default='m', help='Model size (n/s/m) – m fits borderline on 4GB')
    parser.add_argument('--base-epochs', type=int, default=200, help='Base training epochs')
    parser.add_argument('--refine-epochs', type=int, default=50, help='Refinement epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for 4GB GPU')
    parser.add_argument('--device', default='0', help='CUDA device id or cpu')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--mosaic', type=float, default=0.8, help='Mosaic augmentation prob')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial LR')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--amp', type=int, default=0, help='Use AMP (0 disabled defaults for stability)')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset setup step')
    args = parser.parse_args()

    runner = ExperimentRunner()

    overrides = {
        'batch_size': args.batch,
        'image_size': args.imgsz,
        'base_epochs': args.base_epochs,
        'refinement_epochs': args.refine_epochs,
        'lr0': args.lr0,
        'mosaic': args.mosaic,
        'patience': args.patience,
        'mixed_precision': bool(args.amp),
        'save_period': max(args.base_epochs // 10, 10),
        'val_period': 5,
        'workers': args.workers,
        'device': args.device,
    }

    runner.run_refinement_experiment('local_development', data=args.data, model=args.model, **overrides)


if __name__ == '__main__':
    main()
