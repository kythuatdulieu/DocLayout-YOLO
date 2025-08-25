#!/usr/bin/env python3
"""
Enhanced training script for DocLayout-YOLO with refinement module.
Supports two-stage training: base model then refinement module.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from doclayout_yolo import YOLOv10
from doclayout_yolo.models.yolov10.model_refined import YOLOv10Refined


def setup_logging(log_dir: Path):
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{int(time.time())}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_conda_environment():
    """Ensure we're in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'dla':
        print("⚠️  Warning: Not in 'dla' conda environment!")
        print(f"Current environment: {conda_env}")
        print("Please run: conda activate dla")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)


def train_base_model(args, logger):
    """Train the base YOLOv10 model (Stage 1)."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Training Base YOLOv10 Model")
    logger.info("=" * 60)
    
    # Determine model path
    if args.pretrain:
        if args.pretrain == 'coco':
            model_path = f'yolov10{args.model}.pt'
            pretrain_name = 'coco'
        elif 'pt' in args.pretrain:
            model_path = args.pretrain
            pretrain_name = 'custom'
        else:
            raise ValueError("Invalid pretrained model specified!")
    else:
        model_path = f'yolov10{args.model}.yaml'
        pretrain_name = 'None'
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Pretrain: {pretrain_name}")
    
    # Use base YOLOv10 for first stage
    model = YOLOv10(model_path)
    
    # Training parameters
    base_name = f"stage1_yolov10{args.model}_{args.data}_epoch{args.base_epochs}_imgsz{args.image_size}_bs{args.batch_size}_pretrain_{pretrain_name}"
    
    logger.info(f"Training name: {base_name}")
    logger.info(f"Epochs: {args.base_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.image_size}")
    
    # If epochs == 0 we skip actual training but still need a baseline weights file (user must supply via --base-weights later)
    if args.base_epochs > 0:
        results = model.train(
            data=f'{args.data}.yaml',
            epochs=args.base_epochs,
            warmup_epochs=args.warmup_epochs,
            lr0=args.lr0,
            optimizer=args.optimizer,
            momentum=args.momentum,
            imgsz=args.image_size,
            mosaic=args.mosaic,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            amp=bool(args.amp),
            plots=bool(args.plot),
            exist_ok=True,
            val=bool(args.val),
            val_period=args.val_period,
            save_period=args.save_period,
            patience=args.patience,
            project=args.project,
            name=base_name,
        )
    else:
        logger.info("Base epochs = 0 -> Skip base training stage.")
        results = None
    
    # Get path to best model
    base_weights = Path(args.project) / base_name / "weights" / "best.pt"
    
    logger.info(f"Base model training completed!")
    logger.info(f"Best weights saved to: {base_weights}")
    
    return str(base_weights), results


def train_refinement_module(args, base_weights, logger):
    """Train the refinement module (Stage 2)."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Training Refinement Module")
    logger.info("=" * 60)
    
    logger.info(f"Loading base weights from: {base_weights}")
    
    # Use refined model for second stage
    # Accept either weights file or model yaml; if base_weights points to a file use it, else treat as model config
    bw_path = Path(base_weights)
    # Always instantiate model with architecture yaml to ensure cfg is valid
    arch_cfg = f'yolov10{args.model}.yaml'
    model = YOLOv10Refined(arch_cfg)
    # Load weights manually if file exists
    if bw_path.exists() and bw_path.is_file():
        try:
            model.load(str(bw_path))
        except Exception as e:
            logger.warning(f"Failed to load base weights {bw_path}: {e}. Proceeding without.")
    
    # Training parameters for refinement
    refinement_name = f"stage2_refined_{args.data}_epoch{args.refinement_epochs}_imgsz{args.image_size}_bs{args.batch_size}"
    
    logger.info(f"Training name: {refinement_name}")
    logger.info(f"Epochs: {args.refinement_epochs}")
    logger.info(f"Learning rate: {args.refinement_lr}")
    
    # Train refinement module
    results = model.train_refinement(
        base_weights=str(bw_path) if bw_path.exists() else None,
        data=f'{args.data}.yaml',
        epochs=args.refinement_epochs,
        lr0=args.refinement_lr,
        optimizer=args.optimizer,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        amp=bool(args.amp),
        plots=bool(args.plot),
        exist_ok=True,
        val=bool(args.val),
        val_period=args.val_period,
        save_period=args.save_period,
        patience=args.patience // 2,  # Less patience for refinement
        project=args.project,
        name=refinement_name,
    )
    
    # Get path to final model
    final_weights = Path(args.project) / refinement_name / "weights" / "best.pt"
    
    logger.info(f"Refinement training completed!")
    logger.info(f"Final model saved to: {final_weights}")
    
    return str(final_weights), results


def evaluate_model(weights_path, args, logger, stage_name=""):
    """Evaluate model performance."""
    logger.info(f"Evaluating {stage_name} model: {weights_path}")
    
    try:
        if "refined" in weights_path or "stage2" in weights_path:
            model = YOLOv10Refined(weights_path)
        else:
            model = YOLOv10(weights_path)
        
        # Run validation
        results = model.val(
            data=f'{args.data}.yaml',
            batch=args.batch_size,
            device=args.device,
        )
        
        # Log key metrics
        if hasattr(results, 'box'):
            logger.info(f"{stage_name} Results:")
            logger.info(f"  mAP50: {results.box.map50:.4f}")
            logger.info(f"  mAP50-95: {results.box.map:.4f}")
            
            # Per-class results
            if hasattr(results.box, 'mp') and hasattr(results.box, 'mr'):
                logger.info(f"  Precision: {results.box.mp:.4f}")
                logger.info(f"  Recall: {results.box.mr:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Enhanced DocLayout-YOLO training with refinement module")
    
    # Dataset and model
    parser.add_argument('--data', default='doclaynet', required=False, type=str, help='Dataset name')
    parser.add_argument('--model', default='m', required=False, type=str, help='Model size (n/s/m/l/x)')
    parser.add_argument('--pretrain', default=None, required=False, type=str, help='Pretrained weights')
    
    # Training stages
    parser.add_argument('--stage', default='both', choices=['base', 'refinement', 'both'], 
                        help='Training stage to run')
    parser.add_argument('--base-weights', default=None, type=str, 
                        help='Base model weights for refinement stage')
    
    # Training parameters
    parser.add_argument('--base-epochs', default=300, type=int, help='Epochs for base training')
    parser.add_argument('--refinement-epochs', default=100, type=int, help='Epochs for refinement training')
    parser.add_argument('--lr0', default=0.02, type=float, help='Initial learning rate for base training')
    parser.add_argument('--refinement-lr', default=0.001, type=float, help='Learning rate for refinement training')
    parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--warmup-epochs', default=3.0, type=float, help='Warmup epochs')
    
    # Data parameters
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--image-size', default=1120, type=int, help='Image size')
    parser.add_argument('--mosaic', default=1.0, type=float, help='Mosaic augmentation probability')
    
    # Training settings
    parser.add_argument('--workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--device', default="0", type=str, help='Device for training')
    parser.add_argument('--project', default='experiments', type=str, help='Project directory')
    
    # Validation and saving
    parser.add_argument('--val', default=1, type=int, help='Validate during training')
    parser.add_argument('--val-period', default=1, type=int, help='Validation period')
    parser.add_argument('--save-period', default=10, type=int, help='Save period')
    parser.add_argument('--patience', default=100, type=int, help='Early stopping patience')
    parser.add_argument('--plot', default=1, type=int, help='Generate plots')
    parser.add_argument('--amp', default=1, type=int, help='Use Automatic Mixed Precision (1/0)')
    
    # Environment
    parser.add_argument('--skip-env-check', action='store_true', help='Skip conda environment check')
    
    args = parser.parse_args()
    
    # Validate environment
    if not args.skip_env_check:
        validate_conda_environment()
    
    # Setup logging
    log_dir = Path(args.project) / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("DocLayout-YOLO Enhanced Training Started")
    logger.info(f"Training stage: {args.stage}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Model: yolov10{args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Training pipeline
    try:
        if args.stage in ['base', 'both']:
            # Stage 1: Train base model
            base_weights, base_results = train_base_model(args, logger)
            
            # Evaluate base model
            evaluate_model(base_weights, args, logger, "Base")
            
            if args.stage == 'both':
                # Stage 2: Train refinement module
                final_weights, refinement_results = train_refinement_module(args, base_weights, logger)
                
                # Evaluate refined model
                evaluate_model(final_weights, args, logger, "Refined")
                
        elif args.stage == 'refinement':
            if not args.base_weights:
                raise ValueError("--base-weights required for refinement stage")
            
            # Stage 2 only: Train refinement module
            final_weights, refinement_results = train_refinement_module(args, args.base_weights, logger)
            
            # Evaluate refined model
            evaluate_model(final_weights, args, logger, "Refined")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()