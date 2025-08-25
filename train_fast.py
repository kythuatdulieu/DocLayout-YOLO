#!/usr/bin/env python3
"""
Fast Training Script for DocLayout-YOLO with Refinement Module
Optimized for RTX 2050 4GB GPU with 3-hour training time limit

Usage:
    python train_fast.py --model n                    # Base training only (30 mins)
    python train_fast.py --model n --refinement       # Two-stage training (60 mins)
    python train_fast.py --model m --refinement       # Larger model (90 mins)
"""

import os
import argparse
import time
import logging
from pathlib import Path
import yaml

from doclayout_yolo.models.yolov10 import YOLOv10Refined
from doclayout_yolo.utils import LOGGER

def setup_logging(experiment_dir):
    """Setup logging for the training session."""
    log_file = experiment_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_hardware_config(hardware_type="local_development"):
    """Load hardware-specific configuration."""
    config_path = Path(__file__).parent / "configs" / "hardware_configs.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs.get(hardware_type, configs["local_development"])

def train_base_model(model, config, experiment_dir, logger):
    """Train the base YOLOv10 model (Stage 1)."""
    logger.info("🚀 Starting Stage 1: Base Model Training")
    start_time = time.time()
    
    # Training parameters
    train_args = {
        'data': 'doclaynet_mini.yaml',  # Use mini dataset for fast training
        'epochs': config['base_epochs'],
        'batch': config['batch_size'],
        'imgsz': config['image_size'],
        'device': config['device'],
        'workers': config['workers'],
        'lr0': config['lr0'],
        'warmup_epochs': config['warmup_epochs'],
        'mosaic': config['mosaic'],
        'amp': config['mixed_precision'],
        'patience': config['patience'],
        'save_period': config['save_period'],
        'val_period': config['val_period'],
        'project': str(experiment_dir),
        'name': 'base_model',
        'exist_ok': True,
        'plots': True,
        'verbose': True
    }
    
    logger.info(f"Base training parameters: {train_args}")
    
    # Train base model
    results = model.train(**train_args)
    
    # Find best weights
    weights_dir = experiment_dir / "base_model" / "weights"
    best_weights = weights_dir / "best.pt"
    
    training_time = time.time() - start_time
    logger.info(f"✅ Stage 1 completed in {training_time/60:.1f} minutes")
    logger.info(f"📊 Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    logger.info(f"💾 Best weights saved to: {best_weights}")
    
    return str(best_weights), results

def train_refinement_module(model, base_weights, config, experiment_dir, logger):
    """Train the refinement module (Stage 2)."""
    logger.info("🔧 Starting Stage 2: Refinement Module Training")
    start_time = time.time()
    
    # Refinement training parameters
    refinement_args = {
        'base_weights': base_weights,
        'data': 'doclaynet_mini.yaml',
        'epochs': config['refinement_epochs'],
        'batch': config['batch_size'],
        'imgsz': config['image_size'],
        'device': config['device'],
        'workers': config['workers'],
        'lr0': config.get('refinement_lr0', config['lr0'] * 0.2),  # Lower LR for refinement
        'warmup_epochs': config.get('refinement_warmup_epochs', 2),
        'mosaic': config['mosaic'] * 0.5,  # Reduced augmentation
        'amp': config['mixed_precision'],
        'patience': config['patience'] // 2,  # Less patience for refinement
        'save_period': config['save_period'],
        'val_period': config['val_period'],
        'project': str(experiment_dir),
        'name': 'refinement_model',
        'exist_ok': True,
        'plots': True,
        'verbose': True
    }
    
    logger.info(f"Refinement training parameters: {refinement_args}")
    
    # Train refinement module
    results = model.train_refinement(**refinement_args)
    
    # Find best weights
    weights_dir = experiment_dir / "refinement_model" / "weights"
    best_weights = weights_dir / "best.pt"
    
    training_time = time.time() - start_time
    logger.info(f"✅ Stage 2 completed in {training_time/60:.1f} minutes")
    logger.info(f"📊 Refined mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    logger.info(f"💾 Refined weights saved to: {best_weights}")
    
    return str(best_weights), results

def evaluate_model(model_path, config, logger):
    """Quick evaluation of the trained model."""
    logger.info(f"📋 Evaluating model: {model_path}")
    
    # Load model for evaluation
    model = YOLOv10Refined(model_path)
    
    # Quick validation
    results = model.val(
        data='doclaynet_mini.yaml',
        device=config['device'],
        batch=config['batch_size'],
        verbose=False
    )
    
    metrics = results.results_dict
    logger.info(f"📊 Final Results:")
    logger.info(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
    logger.info(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    logger.info(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.3f}")
    logger.info(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Fast DocLayout-YOLO Training")
    parser.add_argument('--model', default='n', choices=['n', 's', 'm'], 
                       help='Model size (n=nano, s=small, m=medium)')
    parser.add_argument('--refinement', action='store_true',
                       help='Enable two-stage training with refinement module')
    parser.add_argument('--hardware', default='local_development',
                       help='Hardware configuration to use')
    parser.add_argument('--data', default='doclaynet_mini',
                       help='Dataset to use')
    parser.add_argument('--experiment-name', default=None,
                       help='Custom experiment name')
    
    args = parser.parse_args()
    
    # Load hardware configuration
    config = load_hardware_config(args.hardware)
    
    # Create experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"fast_train_{args.model}_{timestamp}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir)
    
    logger.info("🎯 DocLayout-YOLO Fast Training Started")
    logger.info(f"💻 Hardware: {args.hardware}")
    logger.info(f"🎮 Model: YOLOv10{args.model}")
    logger.info(f"🔧 Refinement: {'Enabled' if args.refinement else 'Disabled'}")
    logger.info(f"📁 Experiment: {experiment_dir}")
    
    # Initialize model
    model_config = f"yolov10{args.model}.yaml"
    model = YOLOv10Refined(model_config)
    
    total_start_time = time.time()
    
    try:
        # Stage 1: Train base model
        base_weights, base_results = train_base_model(model, config, experiment_dir, logger)
        
        final_weights = base_weights
        final_results = base_results
        
        # Stage 2: Train refinement module (if enabled)
        if args.refinement:
            final_weights, final_results = train_refinement_module(
                model, base_weights, config, experiment_dir, logger
            )
        
        # Final evaluation
        evaluate_model(final_weights, config, logger)
        
        # Training summary
        total_time = time.time() - total_start_time
        logger.info(f"🎉 Training completed successfully!")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"💾 Final model: {final_weights}")
        
        # Save training summary
        summary = {
            'model_size': args.model,
            'refinement_enabled': args.refinement,
            'hardware_config': args.hardware,
            'total_training_time_minutes': total_time / 60,
            'final_weights': final_weights,
            'config_used': config
        }
        
        summary_file = experiment_dir / "training_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"📋 Training summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    return final_weights

if __name__ == "__main__":
    main()