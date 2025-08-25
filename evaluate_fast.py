#!/usr/bin/env python3
"""
Comprehensive evaluation script for DocLayout-YOLO refinement module.
Provides detailed performance analysis including timing, per-class metrics, and comparisons.

Usage:
    python evaluate_fast.py --base experiments/base_model/weights/best.pt --refined experiments/refinement_model/weights/best.pt
    python evaluate_fast.py --model experiments/refinement_model/weights/best.pt --data doclaynet_mini.yaml
"""

import argparse
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from doclayout_yolo.models.yolov10 import YOLOv10Refined


class PerformanceEvaluator:
    """Comprehensive performance evaluator for DocLayout-YOLO models."""
    
    def __init__(self, device='0'):
        self.device = device
        self.results = {}
    
    def evaluate_model(self, model_path: str, data_config: str, model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model_path: Path to model weights
            data_config: Dataset configuration file
            model_name: Name for results tracking
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"\n📊 Evaluating {model_name}: {model_path}")
        
        # Load model
        model = YOLOv10Refined(model_path)
        
        # Timing evaluation
        timing_results = self._evaluate_timing(model, data_config)
        
        # Accuracy evaluation
        accuracy_results = self._evaluate_accuracy(model, data_config)
        
        # Memory evaluation
        memory_results = self._evaluate_memory(model)
        
        # Combine all results
        results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'timing': timing_results,
            'accuracy': accuracy_results,
            'memory': memory_results
        }
        
        self.results[model_name] = results
        return results
    
    def _evaluate_timing(self, model, data_config: str) -> Dict[str, float]:
        """Evaluate inference timing performance."""
        print("⏱️  Evaluating inference timing...")
        
        # Create dummy input for timing
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        model.to(self.device)
        model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Timing measurements
        times = []
        num_iterations = 100
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_iterations) * 1000
        fps = 1.0 / (total_time / num_iterations)
        
        return {
            'avg_inference_time_ms': round(avg_time_ms, 2),
            'fps': round(fps, 1),
            'total_time_s': round(total_time, 3),
            'num_iterations': num_iterations
        }
    
    def _evaluate_accuracy(self, model, data_config: str) -> Dict[str, Any]:
        """Evaluate model accuracy using validation dataset."""
        print("🎯 Evaluating accuracy metrics...")
        
        # Run validation
        results = model.val(
            data=data_config,
            device=self.device,
            verbose=False,
            save_json=False,
            save_hybrid=False,
            plots=False
        )
        
        # Extract metrics
        metrics = results.results_dict
        
        # Parse per-class metrics if available
        per_class_metrics = {}
        if hasattr(results, 'ap_class_index') and hasattr(results, 'ap'):
            class_names = results.names if hasattr(results, 'names') else None
            if class_names:
                for i, class_idx in enumerate(results.ap_class_index):
                    class_name = class_names.get(class_idx, f"class_{class_idx}")
                    per_class_metrics[class_name] = {
                        'ap50': round(results.ap[i, 0], 3) if len(results.ap) > i else 0.0,
                        'ap50_95': round(results.ap[i, :].mean(), 3) if len(results.ap) > i else 0.0
                    }
        
        return {
            'mAP50': round(metrics.get('metrics/mAP50(B)', 0.0), 3),
            'mAP50_95': round(metrics.get('metrics/mAP50-95(B)', 0.0), 3),
            'precision': round(metrics.get('metrics/precision(B)', 0.0), 3),
            'recall': round(metrics.get('metrics/recall(B)', 0.0), 3),
            'per_class': per_class_metrics,
            'num_images': metrics.get('num_images', 0)
        }
    
    def _evaluate_memory(self, model) -> Dict[str, Any]:
        """Evaluate memory usage."""
        print("💾 Evaluating memory usage...")
        
        # Model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size estimation (MB)
        param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        memory_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(param_size_mb, 2)
        }
        
        # GPU memory usage if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            
            memory_info.update({
                'gpu_memory_allocated_mb': round(memory_allocated, 2),
                'gpu_memory_reserved_mb': round(memory_reserved, 2)
            })
        
        return memory_info
    
    def compare_models(self, baseline_results: Dict, refined_results: Dict) -> Dict[str, Any]:
        """Compare baseline and refined model performance."""
        print("\n🔍 Comparing model performance...")
        
        comparison = {
            'accuracy_improvement': {
                'mAP50_delta': refined_results['accuracy']['mAP50'] - baseline_results['accuracy']['mAP50'],
                'mAP50_95_delta': refined_results['accuracy']['mAP50_95'] - baseline_results['accuracy']['mAP50_95'],
                'precision_delta': refined_results['accuracy']['precision'] - baseline_results['accuracy']['precision'],
                'recall_delta': refined_results['accuracy']['recall'] - baseline_results['accuracy']['recall']
            },
            'timing_comparison': {
                'speed_ratio': baseline_results['timing']['fps'] / refined_results['timing']['fps'],
                'latency_increase_ms': refined_results['timing']['avg_inference_time_ms'] - baseline_results['timing']['avg_inference_time_ms'],
                'latency_increase_percent': ((refined_results['timing']['avg_inference_time_ms'] - baseline_results['timing']['avg_inference_time_ms']) / baseline_results['timing']['avg_inference_time_ms']) * 100
            },
            'memory_comparison': {
                'size_increase_mb': refined_results['memory']['model_size_mb'] - baseline_results['memory']['model_size_mb'],
                'param_increase': refined_results['memory']['total_parameters'] - baseline_results['memory']['total_parameters']
            }
        }
        
        # Round numerical values
        for category in comparison.values():
            for key, value in category.items():
                if isinstance(value, float):
                    category[key] = round(value, 3)
        
        return comparison
    
    def print_summary(self, results: Dict[str, Any], comparison: Dict[str, Any] = None):
        """Print comprehensive evaluation summary."""
        print("\n" + "="*60)
        print("📋 EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, model_results in results.items():
            print(f"\n🎮 {model_name.upper()}")
            print(f"   Model: {Path(model_results['model_path']).name}")
            print(f"   mAP50: {model_results['accuracy']['mAP50']:.3f}")
            print(f"   mAP50-95: {model_results['accuracy']['mAP50_95']:.3f}")
            print(f"   Precision: {model_results['accuracy']['precision']:.3f}")
            print(f"   Recall: {model_results['accuracy']['recall']:.3f}")
            print(f"   FPS: {model_results['timing']['fps']:.1f}")
            print(f"   Inference Time: {model_results['timing']['avg_inference_time_ms']:.1f} ms")
            print(f"   Model Size: {model_results['memory']['model_size_mb']:.1f} MB")
            print(f"   Parameters: {model_results['memory']['total_parameters']:,}")
        
        if comparison:
            print(f"\n📈 IMPROVEMENT ANALYSIS")
            acc = comparison['accuracy_improvement']
            timing = comparison['timing_comparison']
            memory = comparison['memory_comparison']
            
            print(f"   ✓ mAP50 improvement: {acc['mAP50_delta']:+.3f}")
            print(f"   ✓ mAP50-95 improvement: {acc['mAP50_95_delta']:+.3f}")
            print(f"   ⚡ Speed impact: {timing['latency_increase_percent']:+.1f}%")
            print(f"   💾 Size increase: {memory['size_increase_mb']:+.1f} MB")
            
            # Performance verdict
            if acc['mAP50_delta'] > 0.005:  # 0.5% improvement threshold
                print(f"   🎉 VERDICT: Refinement shows meaningful improvement!")
            elif acc['mAP50_delta'] > 0:
                print(f"   ⚠️  VERDICT: Marginal improvement, consider optimization")
            else:
                print(f"   ❌ VERDICT: No improvement, check training strategy")
        
        print("="*60)
    
    def save_results(self, output_dir: Path, comparison: Dict[str, Any] = None):
        """Save detailed results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comparison if available
        if comparison:
            comparison_file = output_dir / "model_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
        
        # Save summary
        summary_file = output_dir / "evaluation_summary.yaml"
        summary = {
            'models_evaluated': list(self.results.keys()),
            'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary_metrics': {
                name: {
                    'mAP50': results['accuracy']['mAP50'],
                    'fps': results['timing']['fps'],
                    'model_size_mb': results['memory']['model_size_mb']
                }
                for name, results in self.results.items()
            }
        }
        
        if comparison:
            summary['improvement_analysis'] = comparison['accuracy_improvement']
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"\n💾 Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DocLayout-YOLO Evaluation")
    parser.add_argument('--base', type=str, help='Path to baseline model weights')
    parser.add_argument('--refined', type=str, help='Path to refined model weights')
    parser.add_argument('--model', type=str, help='Path to single model weights (alternative to --base/--refined)')
    parser.add_argument('--data', default='doclaynet_mini.yaml', help='Dataset configuration file')
    parser.add_argument('--device', default='0', help='Device to use for evaluation')
    parser.add_argument('--output', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PerformanceEvaluator(device=args.device)
    
    results = {}
    comparison = None
    
    if args.base and args.refined:
        # Compare two models
        print("🔄 Comparing baseline vs refined models...")
        results['baseline'] = evaluator.evaluate_model(args.base, args.data, "baseline")
        results['refined'] = evaluator.evaluate_model(args.refined, args.data, "refined")
        comparison = evaluator.compare_models(results['baseline'], results['refined'])
        
    elif args.model:
        # Evaluate single model
        print("📊 Evaluating single model...")
        model_name = Path(args.model).stem
        results[model_name] = evaluator.evaluate_model(args.model, args.data, model_name)
        
    else:
        print("❌ Error: Provide either --model OR both --base and --refined")
        return
    
    # Print summary
    evaluator.print_summary(results, comparison)
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("evaluation_results") / time.strftime("%Y%m%d_%H%M%S")
    
    evaluator.save_results(output_dir, comparison)


if __name__ == "__main__":
    main()