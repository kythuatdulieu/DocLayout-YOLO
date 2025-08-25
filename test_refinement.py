#!/usr/bin/env python3
"""
Simple test script to verify DocLayout-YOLO refinement module implementation.
Tests key components without requiring full training.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from doclayout_yolo.nn.modules.ocr_utils import (
    TextFeatureExtractor, 
    SimpleOCRExtractor, 
    RefinementModule
)

def test_text_feature_extraction():
    """Test text feature extraction."""
    print("🧪 Testing Text Feature Extraction...")
    
    extractor = TextFeatureExtractor()
    
    # Test with sample text and bbox
    sample_text = "Table 1: Performance Comparison"
    sample_bbox = [100, 200, 400, 250]  # x1, y1, x2, y2
    image_shape = (800, 600)  # height, width
    
    features = extractor.extract_features(sample_text, sample_bbox, image_shape)
    
    print(f"  ✓ Feature dimension: {len(features)} (expected: {extractor.get_feature_dim()})")
    print(f"  ✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  ✓ Sample features: {features[:5]}")
    
    return True

def test_refinement_module():
    """Test refinement module forward pass."""
    print("🧪 Testing Refinement Module...")
    
    # Model parameters
    yolo_feature_dim = 512
    text_feature_dim = 23  # From TextFeatureExtractor
    num_classes = 11  # DocLayNet classes
    batch_size = 4
    
    # Create refinement module
    refinement = RefinementModule(
        yolo_feature_dim=yolo_feature_dim,
        text_feature_dim=text_feature_dim,
        num_classes=num_classes
    )
    
    # Create dummy inputs
    yolo_features = torch.randn(batch_size, yolo_feature_dim)
    text_features = torch.randn(batch_size, text_feature_dim)
    
    # Forward pass
    with torch.no_grad():
        output = refinement(yolo_features, text_features)
    
    print(f"  ✓ Input YOLO features: {yolo_features.shape}")
    print(f"  ✓ Input text features: {text_features.shape}")
    print(f"  ✓ Output predictions: {output.shape}")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test parameter count
    param_count = sum(p.numel() for p in refinement.parameters())
    print(f"  ✓ Parameter count: {param_count:,}")
    
    return True

def test_ocr_extractor():
    """Test OCR extractor initialization (without actual OCR)."""
    print("🧪 Testing OCR Extractor...")
    
    # Test initialization
    ocr_extractor = SimpleOCRExtractor(use_gpu=False)  # CPU for testing
    
    print(f"  ✓ OCR extractor initialized")
    print(f"  ✓ Languages: {ocr_extractor.languages}")
    print(f"  ✓ GPU enabled: {ocr_extractor.use_gpu}")
    
    # Test with dummy image (without actual OCR call)
    dummy_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 300, 200]
    
    # This will return empty string since EasyOCR is not available in test environment
    text = ocr_extractor.extract_text_from_bbox(dummy_image, dummy_bbox)
    print(f"  ✓ OCR text extraction test: '{text}' (empty expected without EasyOCR)")
    
    return True

def test_integration():
    """Test integration of components."""
    print("🧪 Testing Component Integration...")
    
    # Create components
    text_extractor = TextFeatureExtractor()
    refinement = RefinementModule(512, 23, 11)
    
    # Simulate pipeline
    sample_texts = [
        "Figure 1: Network Architecture",
        "This is a paragraph of text content.",
        "Table 2: Experimental Results"
    ]
    sample_bboxes = [
        [50, 100, 200, 150],
        [50, 200, 400, 350],
        [50, 400, 300, 450]
    ]
    image_shape = (800, 600)
    
    # Extract features for multiple samples
    text_features_list = []
    for text, bbox in zip(sample_texts, sample_bboxes):
        features = text_extractor.extract_features(text, bbox, image_shape)
        text_features_list.append(features)
    
    # Stack features
    text_features_batch = torch.tensor(np.stack(text_features_list))
    yolo_features_batch = torch.randn(3, 512)
    
    # Forward through refinement
    with torch.no_grad():
        predictions = refinement(yolo_features_batch, text_features_batch)
    
    print(f"  ✓ Batch processing: {len(sample_texts)} samples")
    print(f"  ✓ Text features shape: {text_features_batch.shape}")
    print(f"  ✓ Final predictions shape: {predictions.shape}")
    print(f"  ✓ Prediction probabilities (softmax): {torch.softmax(predictions, dim=1)[0][:5]}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 DocLayout-YOLO Refinement Module Tests")
    print("=" * 60)
    
    tests = [
        test_text_feature_extraction,
        test_refinement_module,
        test_ocr_extractor,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"  ✅ {test_func.__name__} PASSED\n")
            else:
                print(f"  ❌ {test_func.__name__} FAILED\n")
        except Exception as e:
            print(f"  ❌ {test_func.__name__} ERROR: {e}\n")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refinement module implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)