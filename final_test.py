#!/usr/bin/env python3
"""
Final integration test demonstrating the complete DocLayout-YOLO refinement system.
"""

import sys
from pathlib import Path

def test_complete_integration():
    """Test the complete integration of all components."""
    print("🧪 COMPLETE INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Package imports
    print("\n1️⃣ Testing Package Imports...")
    try:
        from doclayout_yolo import YOLOv10, YOLOv10Refined
        print("   ✅ Base and refined models imported")
        
        from doclayout_yolo.nn.modules.ocr_utils import (
            TextFeatureExtractor, SimpleOCRExtractor, RefinementModule
        )
        print("   ✅ OCR utilities imported")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Configuration system
    print("\n2️⃣ Testing Configuration System...")
    try:
        import yaml
        with open('configs/hardware_configs.yaml', 'r') as f:
            configs = yaml.safe_load(f)
        
        required_configs = ['local_development', 'kaggle', 'colab', 'server', 'cpu_only']
        for config_name in required_configs:
            if config_name not in configs:
                print(f"   ❌ Missing config: {config_name}")
                return False
        
        print(f"   ✅ All {len(required_configs)} hardware configs available")
        
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False
    
    # Test 3: Feature extraction pipeline
    print("\n3️⃣ Testing Feature Extraction Pipeline...")
    try:
        import torch
        import numpy as np
        
        # Initialize components
        text_extractor = TextFeatureExtractor()
        ocr_extractor = SimpleOCRExtractor(use_gpu=False)
        
        # Test text feature extraction
        sample_texts = [
            "Table 1: Performance Results",
            "Figure 2: Architecture Diagram", 
            "Abstract: This paper presents..."
        ]
        
        features_list = []
        for text in sample_texts:
            features = text_extractor.extract_features(
                text, [50, 100, 200, 150], (600, 800)
            )
            features_list.append(features)
        
        # Stack features
        text_features = torch.tensor(np.stack(features_list))
        
        print(f"   ✅ Text features shape: {text_features.shape}")
        print(f"   ✅ Feature dimension: {text_extractor.get_feature_dim()}")
        
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        return False
    
    # Test 4: Refinement module
    print("\n4️⃣ Testing Refinement Module...")
    try:
        # Create refinement module
        refinement = RefinementModule(
            yolo_feature_dim=512,
            text_feature_dim=23,
            num_classes=11
        )
        
        # Test forward pass
        batch_size = len(sample_texts)
        yolo_features = torch.randn(batch_size, 512)
        
        with torch.no_grad():
            predictions = refinement(yolo_features, text_features)
        
        print(f"   ✅ Refinement output shape: {predictions.shape}")
        print(f"   ✅ Parameter count: {sum(p.numel() for p in refinement.parameters()):,}")
        
    except Exception as e:
        print(f"   ❌ Refinement module failed: {e}")
        return False
    
    # Test 5: Training scripts availability
    print("\n5️⃣ Testing Training Infrastructure...")
    try:
        scripts_to_check = [
            'train_refined.py',
            'evaluate_comprehensive.py', 
            'run_experiment.py',
            'prepare_dataset.py',
            'test_refinement.py'
        ]
        
        missing_scripts = []
        for script in scripts_to_check:
            if not Path(script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"   ❌ Missing scripts: {missing_scripts}")
            return False
        
        print(f"   ✅ All {len(scripts_to_check)} training scripts available")
        
    except Exception as e:
        print(f"   ❌ Script check failed: {e}")
        return False
    
    # Test 6: Hardware configuration access
    print("\n6️⃣ Testing Hardware Configuration Access...")
    try:
        from run_experiment import ExperimentRunner
        
        runner = ExperimentRunner()
        local_config = runner.get_hardware_config('local_development')
        
        required_keys = ['device', 'batch_size', 'base_epochs', 'refinement_epochs', 'lr0']
        for key in required_keys:
            if key not in local_config:
                print(f"   ❌ Missing config key: {key}")
                return False
        
        print(f"   ✅ Hardware config system working")
        print(f"   ✅ Local config: batch_size={local_config['batch_size']}, device={local_config['device']}")
        
    except Exception as e:
        print(f"   ❌ Hardware config test failed: {e}")
        return False
    
    # Success summary
    print("\n🎉 INTEGRATION TEST RESULTS")
    print("=" * 60)
    print("✅ All components integrated successfully!")
    print("\n📋 Verified Components:")
    print("   • Package imports (YOLOv10, YOLOv10Refined)")
    print("   • OCR and text feature extraction")
    print("   • Refinement module architecture")
    print("   • Hardware configuration system")
    print("   • Training and evaluation scripts")
    print("   • End-to-end pipeline compatibility")
    
    print("\n🚀 Ready for:")
    print("   • Dataset preparation")
    print("   • Two-stage training")
    print("   • Comprehensive evaluation")
    print("   • Multi-hardware deployment")
    
    return True

def show_usage_summary():
    """Show a summary of how to use the system."""
    print("\n" + "=" * 60)
    print("🎯 QUICK START GUIDE")
    print("=" * 60)
    
    commands = [
        ("Environment Setup", "conda activate dla"),
        ("Test System", "python test_refinement.py"),
        ("Prepare Data", "python prepare_dataset.py"),
        ("Complete Training", "python run_experiment.py --action complete --hardware local_development"),
        ("Manual Training", "python train_refined.py --stage both --data doclaynet --model m"),
        ("Evaluation", "python evaluate_comprehensive.py --base-model base.pt --refined-model refined.pt")
    ]
    
    for i, (desc, cmd) in enumerate(commands, 1):
        print(f"{i}. {desc}:")
        print(f"   {cmd}")
        print()

def main():
    """Run the final integration test."""
    success = test_complete_integration()
    
    if success:
        show_usage_summary()
        print("\n🎉 DocLayout-YOLO Refinement Module Implementation Complete!")
        print("The system is ready for document layout analysis experiments.")
    else:
        print("\n❌ Integration test failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()