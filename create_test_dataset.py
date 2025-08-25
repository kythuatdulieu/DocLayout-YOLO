#!/usr/bin/env python3
"""
Create a minimal test dataset for quick validation of DocLayout-YOLO refinement module.
This creates synthetic data to test the training pipeline without requiring the full DocLayNet dataset.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml


def create_synthetic_document_image(size=(640, 480), num_objects=3):
    """Create a synthetic document image with text regions."""
    # Create white background
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Add some colored rectangles to simulate document elements
    annotations = []
    
    for i in range(num_objects):
        # Random position and size
        x = np.random.randint(50, size[0] - 150)
        y = np.random.randint(50, size[1] - 100)
        w = np.random.randint(80, 200)
        h = np.random.randint(30, 80)
        
        # Random class (0-10 for DocLayNet classes)
        class_id = np.random.randint(0, 11)
        
        # Draw rectangle
        color = tuple(np.random.randint(200, 255, 3).tolist())
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        
        # Add some text
        text = f"Element_{i}_{class_id}"
        cv2.putText(img, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Convert to YOLO format (normalized coordinates)
        center_x = (x + w/2) / size[0]
        center_y = (y + h/2) / size[1]
        norm_w = w / size[0]
        norm_h = h / size[1]
        
        annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    return img, annotations


def create_test_dataset(output_dir="test_dataset", num_train=20, num_val=5):
    """Create a minimal test dataset."""
    output_path = Path(output_dir)
    
    # Create directory structure
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Create training data
    train_files = []
    for i in range(num_train):
        img, annotations = create_synthetic_document_image()
        
        # Save image
        img_path = output_path / "images" / "train" / f"train_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        train_files.append(str(img_path))
        
        # Save annotations
        label_path = output_path / "labels" / "train" / f"train_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
    
    # Create validation data
    val_files = []
    for i in range(num_val):
        img, annotations = create_synthetic_document_image()
        
        # Save image
        img_path = output_path / "images" / "val" / f"val_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        val_files.append(str(img_path))
        
        # Save annotations
        label_path = output_path / "labels" / "val" / f"val_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
    
    # Create dataset YAML file
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 11,
        'names': ["Caption", "Footnote", "Formula", "List-item", "Page-footer", 
                 "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Test dataset created:")
    print(f"   📁 Location: {output_path.absolute()}")
    print(f"   🖼️  Train images: {num_train}")
    print(f"   🖼️  Val images: {num_val}")
    print(f"   📄 Config: {yaml_path}")
    
    return str(yaml_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test dataset for DocLayout-YOLO")
    parser.add_argument('--output', default='test_dataset', help='Output directory')
    parser.add_argument('--train-size', type=int, default=20, help='Number of training images')
    parser.add_argument('--val-size', type=int, default=5, help='Number of validation images')
    
    args = parser.parse_args()
    
    dataset_yaml = create_test_dataset(args.output, args.train_size, args.val_size)
    print(f"\n🚀 Ready to train with: python train_fast.py --data {dataset_yaml} --model n")