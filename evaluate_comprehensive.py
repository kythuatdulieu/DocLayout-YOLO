#!/usr/bin/env python3
"""
Comprehensive evaluation script for DocLayout-YOLO with refinement module.
Measures AP50, mAP, timing, FPS, and per-class performance.
"""

import argparse
import time
import json
import csv
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List
import logging

from doclayout_yolo import YOLOv10
from doclayout_yolo.models.yolov10.model_refined import YOLOv10Refined


class ModelEvaluator:
    """Comprehensive model evaluator for DocLayout-YOLO."""
    
    def __init__(self, data_config: str, device: str = "0"):
        """
        Initialize evaluator.
        
        Args:
            data_config: Path to data configuration yaml
            device: Device for evaluation
        """
        self.data_config = data_config
        self.device = device
        self.results = {}
    
    def evaluate_model(self, model_path: str, model_type: str = "base", 
                      batch_size: int = 16, verbose: bool = True) -> Dict:
        """
        Evaluate a model comprehensively.
        
        Args:
            model_path: Path to model weights
            model_type: Type of model ('base' or 'refined')
            batch_size: Batch size for evaluation
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} model: {Path(model_path).name}")
        print(f"{'='*60}")
        
        # Load model
        if model_type == "refined":
            model = YOLOv10Refined(model_path)
            model.enable_refinement(True)
        else:
            model = YOLOv10(model_path)
        
        # Performance metrics
        results = {
            'model_path': model_path,
            'model_type': model_type,
            'batch_size': batch_size,
        }
        
        # 1. Standard validation metrics
        print("Running standard validation...")
        val_start = time.time()
        
        val_results = model.val(
            data=self.data_config,
            batch=batch_size,
            device=self.device,
            verbose=verbose
        )
        
        val_time = time.time() - val_start
        
        # Extract metrics
        if hasattr(val_results, 'box'):
            box_results = val_results.box
            results.update({
                'map50': float(box_results.map50),
                'map50_95': float(box_results.map),
                'precision': float(box_results.mp),
                'recall': float(box_results.mr),
                'val_time_total': val_time,
            })
            
            # Per-class metrics
            if hasattr(box_results, 'ap50') and hasattr(box_results, 'ap'):
                results['per_class_ap50'] = [float(x) for x in box_results.ap50]
                results['per_class_ap'] = [float(x) for x in box_results.ap]
            
            # Class names if available
            if hasattr(val_results, 'names'):
                results['class_names'] = val_results.names
        
        # 2. Speed benchmarking
        print("Running speed benchmark...")
        speed_results = self._benchmark_speed(model, batch_size)
        results.update(speed_results)
        
        # 3. Memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
            results['memory_gb'] = memory_used
        
        # Print summary
        self._print_results(results, verbose)
        
        return results
    
    def _benchmark_speed(self, model, batch_size: int, num_runs: int = 100) -> Dict:
        """Benchmark model inference speed."""
        # Prepare dummy data
        device = next(model.model.parameters()).device
        dummy_input = torch.randn(batch_size, 3, 1120, 1120).to(device)
        
        # Warmup
        model.model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model.model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'inference_time_mean': float(times.mean()),
            'inference_time_std': float(times.std()),
            'fps': float(batch_size / times.mean()),
            'fps_per_image': float(1.0 / (times.mean() / batch_size)),
        }
    
    def _print_results(self, results: Dict, verbose: bool = True):
        """Print evaluation results in a formatted way."""
        print(f"\n📊 EVALUATION RESULTS")
        print(f"{'─'*50}")
        
        # Key metrics
        print(f"🎯 Detection Metrics:")
        print(f"   mAP50: {results.get('map50', 0):.4f}")
        print(f"   mAP50-95: {results.get('map50_95', 0):.4f}")
        print(f"   Precision: {results.get('precision', 0):.4f}")
        print(f"   Recall: {results.get('recall', 0):.4f}")
        
        # Speed metrics
        print(f"\n⚡ Speed Metrics:")
        print(f"   Inference time: {results.get('inference_time_mean', 0)*1000:.2f} ± {results.get('inference_time_std', 0)*1000:.2f} ms")
        print(f"   FPS (batch): {results.get('fps', 0):.1f}")
        print(f"   FPS (per image): {results.get('fps_per_image', 0):.1f}")
        
        # Memory
        if 'memory_gb' in results:
            print(f"   Memory usage: {results['memory_gb']:.2f} GB")
        
        # Per-class results (if verbose)
        if verbose and 'per_class_ap50' in results:
            print(f"\n📋 Per-Class AP50:")
            class_names = results.get('class_names', {})
            for i, ap50 in enumerate(results['per_class_ap50']):
                class_name = class_names.get(i, f"Class_{i}")
                print(f"   {class_name}: {ap50:.4f}")
    
    def compare_models(self, base_path: str, refined_path: str, batch_size: int = 16) -> Dict:
        """Compare base and refined models."""
        print(f"\n🔬 MODEL COMPARISON")
        print(f"{'='*60}")
        
        # Evaluate both models
        base_results = self.evaluate_model(base_path, "base", batch_size)
        refined_results = self.evaluate_model(refined_path, "refined", batch_size)
        
        # Calculate improvements
        comparison = {
            'base': base_results,
            'refined': refined_results,
            'improvements': {}
        }
        
        # Metric improvements
        metrics_to_compare = ['map50', 'map50_95', 'precision', 'recall']
        
        for metric in metrics_to_compare:
            base_val = base_results.get(metric, 0)
            refined_val = refined_results.get(metric, 0)
            improvement = ((refined_val - base_val) / base_val * 100) if base_val > 0 else 0
            comparison['improvements'][metric] = improvement
        
        # Speed comparison
        base_fps = base_results.get('fps_per_image', 0)
        refined_fps = refined_results.get('fps_per_image', 0)
        speed_change = ((refined_fps - base_fps) / base_fps * 100) if base_fps > 0 else 0
        comparison['improvements']['speed'] = speed_change
        
        # Print comparison
        print(f"\n📈 IMPROVEMENT SUMMARY")
        print(f"{'─'*50}")
        for metric, improvement in comparison['improvements'].items():
            direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            print(f"   {metric.upper()}: {improvement:+.2f}% {direction}")
        
        return comparison
    
    def save_results(self, results: Dict, output_dir: str = "evaluation_results"):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save as JSON
        json_path = output_path / f"results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV (flattened)
        csv_path = output_path / f"results_{timestamp}.csv"
        if isinstance(results, dict) and 'base' in results and 'refined' in results:
            # Comparison results
            flattened = []
            for model_type in ['base', 'refined']:
                row = {'model_type': model_type}
                row.update(results[model_type])
                flattened.append(row)
            
            if flattened:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened)
        else:
            # Single model results
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
                writer.writerow(results)
        
        print(f"\n💾 Results saved to:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DocLayout-YOLO evaluation")
    
    # Model and data
    parser.add_argument('--data', default='doclaynet.yaml', help='Data configuration file')
    parser.add_argument('--base-model', default=None, help='Path to base model weights')
    parser.add_argument('--refined-model', default=None, help='Path to refined model weights')
    parser.add_argument('--model', default=None, help='Single model to evaluate')
    parser.add_argument('--model-type', default='base', choices=['base', 'refined'], 
                        help='Type of single model')
    
    # Evaluation settings
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--device', default='0', help='Device for evaluation')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Benchmarking
    parser.add_argument('--speed-runs', default=100, type=int, help='Number of runs for speed benchmark')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model and not (args.base_model and args.refined_model):
        raise ValueError("Either --model or both --base-model and --refined-model must be specified")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.data, args.device)
    
    if args.model:
        # Single model evaluation
        print(f"Evaluating single model: {args.model}")
        results = evaluator.evaluate_model(
            args.model, 
            args.model_type, 
            args.batch_size, 
            args.verbose
        )
        evaluator.save_results(results, args.output_dir)
        
    else:
        # Model comparison
        print(f"Comparing models:")
        print(f"  Base: {args.base_model}")
        print(f"  Refined: {args.refined_model}")
        
        comparison = evaluator.compare_models(
            args.base_model, 
            args.refined_model, 
            args.batch_size
        )
        evaluator.save_results(comparison, args.output_dir)


if __name__ == "__main__":
    main()