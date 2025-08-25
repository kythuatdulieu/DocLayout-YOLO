# DocLayout-YOLO Refinement Module

This implementation extends the original DocLayout-YOLO with a lightweight refinement module that uses text-based semantic features to improve document layout analysis accuracy.

## 🎯 Research Objectives

- Build a lightweight refinement module using simple text features (no large language models)
- Integrate the module with YOLOv10 for enhanced document layout detection
- Implement two-stage training: base model then refinement module
- Evaluate improvements on DocLayNet dataset with comprehensive metrics
- Design for scalability across different hardware configurations

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Enhanced YOLOv10 Architecture               │
├─────────────────────────────────────────────────────────────────┤
│  Input Image                                                    │
│       ↓                                                         │
│  YOLOv10 Backbone → Visual Features                            │
│       ↓                     ↓                                   │
│  Detected Regions → OCR → Text Features                        │
│       ↓                     ↓                                   │
│  Visual Features + Text Features → Refinement Module (MLP)      │
│       ↓                                                         │
│  Enhanced Predictions                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Base YOLOv10 Model**: Standard visual-only document layout detection
2. **OCR Extractor**: Extracts text from detected bounding boxes using EasyOCR
3. **Text Feature Extractor**: Creates lightweight semantic features:
   - Text statistics (length, word count, digit ratio, etc.)
   - Keyword presence for each document class
   - Spatial position features
4. **Refinement Module**: MLP that fuses visual and text features

## 🚀 Quick Start

### Environment Setup

```bash
# Always use the 'dla' conda environment
conda activate dla

# Or set up the environment from scratch
./setup_conda_env.sh
```

### Dataset Preparation

```bash
# Download and prepare DocLayNet dataset
python prepare_dataset.py --data-dir ./layout_data
```

### Training

#### Option 1: Complete Pipeline (Recommended)
```bash
# Run complete two-stage training with hardware-optimized settings
python run_experiment.py --action complete --hardware local_development --model m
```

#### Option 2: Manual Two-Stage Training
```bash
# Stage 1: Train base YOLOv10 model
python train_refined.py --stage base --data doclaynet --model m --base-epochs 300

# Stage 2: Train refinement module (using best weights from stage 1)
python train_refined.py --stage refinement --data doclaynet --model m --refinement-epochs 100 --base-weights experiments/stage1_*/weights/best.pt
```

### Evaluation

```bash
# Comprehensive evaluation with timing and per-class metrics
python evaluate_comprehensive.py --base-model path/to/base/model.pt --refined-model path/to/refined/model.pt --data doclaynet.yaml
```

## ⚙️ Hardware Configurations

The system includes pre-configured settings for different hardware setups:

| Configuration | Hardware | Batch Size | Epochs | Optimizations |
|---------------|----------|------------|--------|---------------|
| `local_development` | i5-13420H + GTX 2050 5GB | 8 | 200+50 | Memory-efficient |
| `kaggle` | Kaggle GPU | 16 | 300+100 | Time-optimized |
| `colab` | Google Colab | 12 | 300+100 | Session-aware |
| `server` | Multi-GPU Server | 64 | 500+200 | High-performance |
| `cpu_only` | CPU fallback | 2 | 50+20 | CPU-optimized |

### Using Different Configurations

```bash
# For Kaggle environment
python run_experiment.py --hardware kaggle --model m

# For Google Colab
python run_experiment.py --hardware colab --model s

# For high-end server
python run_experiment.py --hardware server --model l
```

## 📊 Evaluation Metrics

The comprehensive evaluation includes:

- **Detection Metrics**: mAP50, mAP50-95, Precision, Recall
- **Speed Metrics**: Inference time, FPS (batch and per-image)
- **Memory Usage**: Peak GPU memory consumption
- **Per-Class Analysis**: AP50 for each document element type
- **Improvement Analysis**: Quantified gains from refinement module

## 🔬 Experimental Design

### Two-Stage Training Strategy

1. **Stage 1 - Base Training**:
   - Train YOLOv10 backbone with standard visual features
   - Use standard YOLO loss functions
   - Save best model based on mAP50

2. **Stage 2 - Refinement Training**:
   - Load pre-trained base model
   - Freeze all YOLO parameters
   - Train only refinement module parameters
   - Use combined loss (base + refinement predictions)

### Text Feature Engineering

The refinement module uses lightweight text features extracted from OCR:

1. **Statistical Features**:
   - Character/word count
   - Digit ratio, uppercase ratio
   - Special character count

2. **Semantic Features**:
   - Keyword presence for each DocLayNet class
   - Class-specific scoring based on content

3. **Spatial Features**:
   - Normalized position (center, size)
   - Relative location on page

## 📁 File Structure

```
DocLayout-YOLO/
├── configs/
│   └── hardware_configs.yaml          # Hardware-specific configurations
├── doclayout_yolo/
│   ├── nn/modules/
│   │   └── ocr_utils.py               # OCR and text feature extraction
│   └── models/yolov10/
│       └── model_refined.py           # Enhanced YOLOv10 with refinement
├── prepare_dataset.py                 # Dataset preparation script
├── train_refined.py                   # Two-stage training script
├── evaluate_comprehensive.py          # Comprehensive evaluation
├── run_experiment.py                  # Main experiment runner
├── setup_conda_env.sh                # Environment setup script
└── README_REFINEMENT.md              # This file
```

## 📋 Experiment Checklist

- [x] Environment setup with conda 'dla' environment
- [x] Dataset preparation (DocLayNet)
- [x] OCR integration module
- [x] Text feature extraction
- [x] Refinement module (MLP) implementation
- [x] Two-stage training pipeline
- [x] Hardware-specific configurations
- [x] Comprehensive evaluation system
- [x] Scalable design for different platforms

## 🎛️ Configuration Options

### Training Parameters

```yaml
# Example configuration for local development
local_development:
  device: "0"
  batch_size: 8
  workers: 4
  image_size: 1120
  base_epochs: 200
  refinement_epochs: 50
  lr0: 0.01
  refinement_lr: 0.001
  patience: 50
```

### Key Parameters

- `base_epochs`: Training epochs for Stage 1 (base model)
- `refinement_epochs`: Training epochs for Stage 2 (refinement module)
- `refinement_lr`: Learning rate for refinement training (typically lower)
- `batch_size`: Adjusted for hardware memory constraints
- `image_size`: Input image resolution (1120 for DocLayNet)

## 🚨 Important Notes

1. **Always activate conda environment**: Use `conda activate dla` before running any scripts
2. **Memory management**: Batch sizes are pre-configured for different hardware
3. **Two-stage approach**: Base model must be trained before refinement
4. **OCR dependency**: Requires EasyOCR for text extraction
5. **Dataset path**: Ensure DocLayNet is downloaded and configured correctly

## 📈 Expected Results

Based on the research methodology, the refinement module should provide:

- **Improved mAP**: 1-3% improvement over base YOLOv10
- **Better class-specific performance**: Particularly for text-heavy classes
- **Minimal speed impact**: <10% inference time increase
- **Scalable training**: Faster refinement stage compared to full retraining

## 🔧 Troubleshooting

### Common Issues

1. **OCR not working**: Install EasyOCR with `pip install easyocr`
2. **CUDA out of memory**: Reduce batch size in hardware config
3. **Dataset not found**: Run `python prepare_dataset.py` first
4. **Wrong conda environment**: Ensure `conda activate dla` is executed

### Performance Optimization

- Use mixed precision training on supported hardware
- Adjust batch size based on available GPU memory
- Use appropriate number of workers for your CPU cores
- Enable gradient checkpointing for very large models

## 📚 References

- [DocLayout-YOLO Paper](https://arxiv.org/abs/2410.12628)
- [YOLOv10 Architecture](https://github.com/THU-MIG/yolov10)
- [DocLayNet Dataset](https://github.com/DS4SD/DocLayNet)