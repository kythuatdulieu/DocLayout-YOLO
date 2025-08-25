#!/usr/bin/env python3
"""
Dataset preparation script for DocLayout-YOLO refinement module experiments.
Downloads and prepares DocLayNet dataset.
"""

import os
import sys
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

def download_doclaynet(data_dir="./layout_data"):
    """Download DocLayNet dataset from HuggingFace."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading DocLayNet dataset to {data_path.absolute()}...")
    
    try:
        snapshot_download(
            repo_id="juliozhao/doclayout-yolo-DocLayNet",
            local_dir=str(data_path / "doclaynet"),
            repo_type="dataset"
        )
        print("DocLayNet dataset downloaded successfully!")
        
        # Verify dataset structure
        doclaynet_path = data_path / "doclaynet"
        required_files = ["train.txt", "val.txt", "images", "labels"]
        
        for file in required_files:
            if not (doclaynet_path / file).exists():
                print(f"Warning: {file} not found in dataset")
            else:
                print(f"✓ Found {file}")
                
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def verify_dataset_config():
    """Verify dataset configuration in ultralytics settings."""
    try:
        import doclayout_yolo
        from doclayout_yolo.utils import SETTINGS
        
        current_datasets_dir = SETTINGS.get('datasets_dir', '')
        project_root = Path(__file__).parent.absolute()
        
        print(f"Current datasets_dir: {current_datasets_dir}")
        print(f"Project root: {project_root}")
        
        if str(project_root) not in current_datasets_dir:
            print(f"⚠️  Consider updating datasets_dir in ultralytics settings to: {project_root}")
            print("You can find the settings file typically at: $HOME/.config/Ultralytics/settings.yaml")
        else:
            print("✓ Dataset directory configuration looks good")
            
    except Exception as e:
        print(f"Could not verify dataset configuration: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for DocLayout-YOLO refinement experiments")
    parser.add_argument("--data-dir", default="./layout_data", help="Directory to store datasets")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    
    args = parser.parse_args()
    
    print("DocLayout-YOLO Dataset Preparation")
    print("=" * 40)
    
    if not args.skip_download:
        success = download_doclaynet(args.data_dir)
        if not success:
            sys.exit(1)
    
    verify_dataset_config()
    
    print("\n" + "=" * 40)
    print("Dataset preparation complete!")
    print(f"Dataset location: {Path(args.data_dir).absolute()}")
    print("\nNext steps:")
    print("1. Ensure conda environment 'dla' is activated")
    print("2. Run baseline training with: python train.py --data doclaynet --model m-doclayout --epoch 100")

if __name__ == "__main__":
    main()