#!/usr/bin/env python3
"""
Main experiment runner for DocLayout-YOLO refinement module research.
Handles configuration loading, environment setup, and complete experiment pipeline.
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class ExperimentRunner:
    """Main experiment runner for DocLayout-YOLO refinement research."""
    
    def __init__(self, config_file: str = "configs/hardware_configs.yaml"):
        """
        Initialize experiment runner.
        
        Args:
            config_file: Path to hardware configuration file
        """
        self.project_root = PROJECT_ROOT
        self.config_file = self.project_root / config_file
        self.configs = self._load_configs()
        self.logger = self._setup_logging()
        
    def _load_configs(self) -> Dict[str, Any]:
        """Load hardware configurations."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.project_root / 'experiment.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def check_environment(self) -> bool:
        """Check if the environment is properly set up."""
        self.logger.info("Checking environment setup...")
        
        # Check conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        if conda_env != 'dla':
            self.logger.warning(f"Not in 'dla' conda environment (current: {conda_env})")
            return False
        
        # Check required packages
        try:
            import torch
            import doclayout_yolo
            self.logger.info(f"✓ PyTorch version: {torch.__version__}")
            self.logger.info(f"✓ DocLayout-YOLO version: {doclayout_yolo.__version__}")
            self.logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                self.logger.info(f"✓ CUDA device: {torch.cuda.get_device_name()}")
                self.logger.info(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
        except ImportError as e:
            self.logger.error(f"Missing required package: {e}")
            return False
        
        return True
    
    def setup_dataset(self, skip_download: bool = False) -> bool:
        """Set up dataset for experiments."""
        self.logger.info("Setting up dataset...")
        
        script_path = self.project_root / "prepare_dataset.py"
        cmd = ["python", str(script_path)]
        
        if skip_download:
            cmd.append("--skip-download")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("Dataset setup completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dataset setup failed: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            return False
    
    def get_hardware_config(self, hardware_type: str) -> Dict[str, Any]:
        """Get configuration for specific hardware type."""
        if hardware_type not in self.configs:
            available = list(self.configs.keys())
            raise ValueError(f"Hardware type '{hardware_type}' not found. Available: {available}")
        
        return self.configs[hardware_type]
    
    def run_baseline_experiment(self, hardware_type: str, data: str = "doclaynet", 
                               model: str = "m", **overrides) -> bool:
        """Run baseline experiment."""
        self.logger.info(f"Running baseline experiment with {hardware_type} configuration")
        
        config = self.get_hardware_config(hardware_type)
        config.update(overrides)  # Apply any overrides
        
        # Build command
        cmd = [
            "python", str(self.project_root / "train.py"),
            "--data", data,
            "--model", model,
            "--epoch", str(config.get("base_epochs", 300)),
            "--batch-size", str(config["batch_size"]),
            "--image-size", str(config["image_size"]),
            "--device", config["device"],
            "--workers", str(config["workers"]),
            "--lr0", str(config["lr0"]),
            "--warmup-epochs", str(config["warmup_epochs"]),
            "--mosaic", str(config["mosaic"]),
            "--patience", str(config["patience"]),
            "--save-period", str(config["save_period"]),
            "--val-period", str(config["val_period"]),
            "--project", "experiments/baseline",
            "--plot", "1"
        ]

        # Optional AMP flag if script supports it
        if "mixed_precision" in config:
            cmd.extend(["--amp", "1" if config["mixed_precision"] else "0"])  # train.py should ignore if unsupported
        
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            self.logger.info("Baseline experiment completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Baseline experiment failed: {e}")
            return False
    
    def run_refinement_experiment(self, hardware_type: str, base_weights: str = None,
                                 data: str = "doclaynet", model: str = "m", **overrides) -> bool:
        """Run refinement experiment."""
        self.logger.info(f"Running refinement experiment with {hardware_type} configuration")
        
        config = self.get_hardware_config(hardware_type)
        config.update(overrides)  # Apply any overrides
        
        # Build command
        cmd = [
            "python", str(self.project_root / "train_refined.py"),
            "--data", data,
            "--model", model,
            "--stage", "both" if not base_weights else "refinement",
            "--base-epochs", str(config.get("base_epochs", 300)),
            "--refinement-epochs", str(config.get("refinement_epochs", 100)),
            "--batch-size", str(config["batch_size"]),
            "--image-size", str(config["image_size"]),
            "--device", config["device"],
            "--workers", str(config["workers"]),
            "--lr0", str(config["lr0"]),
            "--refinement-lr", str(config.get("refinement_lr", config["lr0"] * 0.1)),
            "--warmup-epochs", str(config["warmup_epochs"]),
            "--patience", str(config["patience"]),
            "--save-period", str(config["save_period"]),
            "--val-period", str(config["val_period"]),
            "--project", "experiments/refinement",
            "--plot", "1"
        ]

        # Add AMP flag for refined training
        if "mixed_precision" in config:
            cmd.extend(["--amp", "1" if config["mixed_precision"] else "0"])
        
        if base_weights:
            cmd.extend(["--base-weights", base_weights])
        
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            self.logger.info("Refinement experiment completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Refinement experiment failed: {e}")
            return False
    
    def run_evaluation(self, base_model: str = None, refined_model: str = None,
                      single_model: str = None, hardware_type: str = "local_development") -> bool:
        """Run comprehensive evaluation."""
        self.logger.info("Running comprehensive evaluation")
        
        config = self.get_hardware_config(hardware_type)
        
        cmd = [
            "python", str(self.project_root / "evaluate_comprehensive.py"),
            "--data", "doclaynet.yaml",
            "--batch-size", str(config["batch_size"]),
            "--device", config["device"],
            "--verbose"
        ]
        
        if single_model:
            cmd.extend(["--model", single_model])
        elif base_model and refined_model:
            cmd.extend(["--base-model", base_model, "--refined-model", refined_model])
        else:
            self.logger.error("Either single_model or both base_model and refined_model must be specified")
            return False
        
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            self.logger.info("Evaluation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Evaluation failed: {e}")
            return False
    
    def run_complete_experiment(self, hardware_type: str, data: str = "doclaynet", 
                               model: str = "m", skip_dataset: bool = False) -> bool:
        """Run complete experiment pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE DOCLAYOUT-YOLO REFINEMENT EXPERIMENT")
        self.logger.info("=" * 80)
        
        # Step 1: Environment check
        if not self.check_environment():
            self.logger.error("Environment check failed")
            return False
        
        # Step 2: Dataset setup
        if not skip_dataset:
            if not self.setup_dataset():
                self.logger.error("Dataset setup failed")
                return False
        
        # Step 3: Run refinement experiment (includes baseline)
        if not self.run_refinement_experiment(hardware_type, data=data, model=model):
            self.logger.error("Refinement experiment failed")
            return False
        
        # Step 4: Find model weights for evaluation
        experiments_dir = self.project_root / "experiments" / "refinement"
        
        # Look for latest weights
        weight_files = list(experiments_dir.glob("**/weights/best.pt"))
        if len(weight_files) >= 2:
            # Assume the last two are base and refined
            weight_files.sort(key=lambda x: x.stat().st_mtime)
            base_weights = str(weight_files[-2])
            refined_weights = str(weight_files[-1])
            
            # Step 5: Run evaluation
            if not self.run_evaluation(base_weights, refined_weights, hardware_type=hardware_type):
                self.logger.error("Evaluation failed")
                return False
        else:
            self.logger.warning("Could not find model weights for evaluation")
        
        self.logger.info("=" * 80)
        self.logger.info("COMPLETE EXPERIMENT FINISHED SUCCESSFULLY")
        self.logger.info("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="DocLayout-YOLO Refinement Experiment Runner")
    
    # Main operation
    parser.add_argument('--action', default='complete', 
                        choices=['complete', 'baseline', 'refinement', 'evaluate', 'setup'],
                        help='Action to perform')
    
    # Hardware configuration
    parser.add_argument('--hardware', default='local_development',
                        choices=['local_development', 'kaggle', 'colab', 'server', 'cpu_only'],
                        help='Hardware configuration to use')
    
    # Dataset and model
    parser.add_argument('--data', default='doclaynet', help='Dataset to use')
    parser.add_argument('--model', default='m', help='Model size (n/s/m/l/x)')
    
    # For evaluation
    parser.add_argument('--base-weights', default=None, help='Base model weights for evaluation')
    parser.add_argument('--refined-weights', default=None, help='Refined model weights for evaluation')
    parser.add_argument('--single-model', default=None, help='Single model for evaluation')
    
    # Options
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset setup')
    parser.add_argument('--config-file', default='configs/hardware_configs.yaml', 
                        help='Hardware configuration file')
    
    args = parser.parse_args()
    
    # Initialize runner
    try:
        runner = ExperimentRunner(args.config_file)
    except Exception as e:
        print(f"Failed to initialize experiment runner: {e}")
        sys.exit(1)
    
    # Execute action
    success = False
    
    if args.action == 'setup':
        success = runner.setup_dataset(skip_download=args.skip_dataset)
        
    elif args.action == 'baseline':
        success = runner.run_baseline_experiment(args.hardware, args.data, args.model)
        
    elif args.action == 'refinement':
        success = runner.run_refinement_experiment(args.hardware, args.base_weights, args.data, args.model)
        
    elif args.action == 'evaluate':
        success = runner.run_evaluation(args.base_weights, args.refined_weights, 
                                       args.single_model, args.hardware)
        
    elif args.action == 'complete':
        success = runner.run_complete_experiment(args.hardware, args.data, args.model, args.skip_dataset)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()