#!/usr/bin/env python3
"""
Quick usage example for DocLayout-YOLO refinement module.
Demonstrates the complete workflow from setup to evaluation.
"""

import os
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Print a step description."""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def main():
    print_header("DocLayout-YOLO Refinement Module Usage Example")
    
    print("""
This example demonstrates how to use the DocLayout-YOLO refinement module
for document layout analysis with text-based semantic enhancement.

The implementation follows the research methodology:
1. Two-stage training (base model → refinement module)
2. Lightweight text features (no large language models)
3. Scalable configuration for different hardware setups
4. Comprehensive evaluation with multiple metrics
    """)
    
    print_step(1, "Environment Setup")
    print("""
# Always use the 'dla' conda environment
conda activate dla

# Or set up from scratch
./setup_conda_env.sh
    """)
    
    print_step(2, "Dataset Preparation")
    print("""
# Download and prepare DocLayNet dataset
python prepare_dataset.py --data-dir ./layout_data

# This will:
# - Download DocLayNet from HuggingFace
# - Verify dataset structure
# - Check configuration paths
    """)
    
    print_step(3, "Quick Test")
    print("""
# Test the refinement module implementation
python test_refinement.py

# This validates:
# - Text feature extraction (23-dim features)
# - Refinement module architecture (~171K params)
# - OCR integration
# - End-to-end pipeline
    """)
    
    print_step(4, "Training Options")
    print("""
# Option A: Complete automated pipeline (RECOMMENDED)
python run_experiment.py --action complete --hardware local_development --model m

# Option B: Manual two-stage training
# Stage 1: Train base YOLOv10
python train_refined.py --stage base --data doclaynet --model m --base-epochs 200

# Stage 2: Train refinement module
python train_refined.py --stage refinement --data doclaynet --model m \\
    --refinement-epochs 50 --base-weights experiments/stage1_*/weights/best.pt
    """)
    
    print_step(5, "Hardware-Specific Configurations")
    print("""
# For GTX 2050 5GB (target hardware)
python run_experiment.py --hardware local_development --model m

# For Kaggle
python run_experiment.py --hardware kaggle --model m

# For Google Colab  
python run_experiment.py --hardware colab --model s

# For high-end server
python run_experiment.py --hardware server --model l

# Configuration includes optimized:
# - Batch sizes, learning rates, epochs
# - Memory management, training schedules
# - Device settings, worker counts
    """)
    
    print_step(6, "Evaluation")
    print("""
# Comprehensive evaluation with metrics
python evaluate_comprehensive.py \\
    --base-model path/to/base/model.pt \\
    --refined-model path/to/refined/model.pt \\
    --data doclaynet.yaml

# Metrics included:
# - Detection: mAP50, mAP50-95, Precision, Recall
# - Speed: Inference time, FPS (batch and per-image)
# - Memory: Peak GPU usage
# - Per-class: AP50 for each document element
# - Improvements: Quantified gains from refinement
    """)
    
    print_step(7, "Key Features")
    print("""
✅ Two-stage training strategy
✅ Lightweight text features (no LLMs)
✅ Hardware-optimized configurations
✅ Comprehensive evaluation metrics
✅ Scalable design (CPU to multi-GPU)
✅ Conda environment integration
✅ Modular, testable components
✅ Ready for Kaggle/Colab deployment
    """)
    
    print_header("Architecture Overview")
    print("""
    Input Image
         ↓
    YOLOv10 Backbone → Visual Features (512-dim)
         ↓                    ↓
    Detected Regions → OCR → Text Features (23-dim)
         ↓                    ↓
    Visual + Text Features → Refinement Module (MLP)
         ↓
    Enhanced Predictions (11 DocLayNet classes)
    
    Key Components:
    - Base YOLOv10: Standard visual detection
    - OCR Extractor: EasyOCR for text extraction  
    - Text Features: Statistics + keywords + spatial
    - Refinement: Lightweight MLP fusion (~171K params)
    """)
    
    print_header("Expected Results")
    print("""
    Based on research methodology, the refinement module should provide:
    
    📈 Performance Improvements:
    - mAP50: +1-3% over base YOLOv10
    - Better performance on text-heavy classes
    - Improved semantic understanding
    
    ⚡ Efficiency:
    - <10% inference time increase
    - Minimal memory overhead
    - Fast refinement training stage
    
    🎯 Research Validation:
    - Demonstrates effectiveness of lightweight semantic features
    - Shows benefits of two-stage training approach
    - Validates modular architecture design
    """)
    
    print_header("Get Started")
    print("""
    1. conda activate dla
    2. python prepare_dataset.py
    3. python test_refinement.py
    4. python run_experiment.py --action complete --hardware local_development
    
    For detailed information, see README_REFINEMENT.md
    """)

if __name__ == "__main__":
    main()