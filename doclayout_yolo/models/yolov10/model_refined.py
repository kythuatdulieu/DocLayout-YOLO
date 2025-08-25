"""
Enhanced YOLOv10 model with refinement module for document layout analysis.
Integrates text-based semantic features with visual features for improved accuracy.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

from doclayout_yolo.engine.model import Model
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
from doclayout_yolo.nn.modules.ocr_utils import (
    SimpleOCRExtractor, 
    TextFeatureExtractor, 
    RefinementModule
)
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer

from huggingface_hub import PyTorchModelHubMixin


class YOLOv10RefinedDetectionModel(YOLOv10DetectionModel):
    """
    YOLOv10 detection model with refinement module.
    Adds text-based semantic features to improve layout detection accuracy.
    """
    
    def __init__(self, cfg="yolov10m.yaml", ch=3, nc=None, verbose=True):
        # Initialize training stage and refinement settings first
        self.training_stage = 'base'
        self.use_refinement = False
        self.refinement_module = None
        
        # Initialize parent class
        super().__init__(cfg, ch, nc, verbose)
        
        # Initialize OCR and text feature extractors
        self.ocr_extractor = SimpleOCRExtractor()
        self.text_feature_extractor = TextFeatureExtractor()
    
    def setup_refinement_module(self, yolo_feature_dim: int = 512, hidden_dims: List[int] = [256, 128]):
        """
        Set up the refinement module after model construction.
        
        Args:
            yolo_feature_dim: Dimension of YOLO features to use for refinement
            hidden_dims: Hidden layer dimensions for refinement MLP
        """
        text_feature_dim = self.text_feature_extractor.get_feature_dim()
        
        self.refinement_module = RefinementModule(
            yolo_feature_dim=yolo_feature_dim,
            text_feature_dim=text_feature_dim,
            num_classes=self.nc,
            hidden_dims=hidden_dims
        )
        
        print(f"Refinement module initialized:")
        print(f"  YOLO feature dim: {yolo_feature_dim}")
        print(f"  Text feature dim: {text_feature_dim}")
        print(f"  Output classes: {self.nc}")
        print(f"  Hidden dims: {hidden_dims}")
    
    def enable_refinement(self, enable: bool = True):
        """Enable or disable refinement module during inference."""
        self.use_refinement = enable and self.refinement_module is not None
        
    def set_training_stage(self, stage: str):
        """
        Set training stage: 'base' for training base YOLO, 'refinement' for training refinement module.
        
        Args:
            stage: 'base' or 'refinement'
        """
        if stage not in ['base', 'refinement']:
            raise ValueError("Training stage must be 'base' or 'refinement'")
        
        self.training_stage = stage
        
        if stage == 'refinement':
            # Freeze YOLO parameters
            for name, param in self.named_parameters():
                if 'refinement_module' not in name:
                    param.requires_grad = False
            print("Froze YOLO parameters for refinement training")
        else:
            # Unfreeze YOLO parameters
            for param in self.parameters():
                param.requires_grad = True
            print("Unfroze YOLO parameters for base training")
    
    def extract_yolo_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from YOLO backbone for refinement.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, features) where features can be used by refinement module
        """
        # Forward through backbone
        for i, layer in enumerate(self.model):
            x = layer(x)
            
            # Extract features from a middle layer for refinement
            # This is a simplified approach - in practice, you might want to 
            # extract features from specific layers
            if i == len(self.model) - 2:  # Second to last layer
                yolo_features = x
        
        predictions = x
        
        return predictions, yolo_features
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Forward pass with optional refinement.
        
        Args:
            x: Input tensor or dict with 'images' and 'text_features' keys
            augment: Whether to use augmentation during inference
            profile: Whether to profile the model
            visualize: Whether to visualize features
            
        Returns:
            Model predictions, potentially refined with text features
        """
        # Handle different input formats
        if isinstance(x, dict):
            images = x['images']
            text_features = x.get('text_features', None)
        else:
            images = x
            text_features = None
        
        # Base YOLO forward pass
        if self.training_stage == 'base' or not self.use_refinement:
            return super().forward(images, augment, profile, visualize)
        
        # Refinement stage forward pass
        predictions, yolo_features = self.extract_yolo_features(images)
        
        if self.use_refinement and self.refinement_module is not None and text_features is not None:
            # Apply refinement
            refined_predictions = self.refinement_module(yolo_features, text_features)
            
            # Combine or replace predictions as needed
            # This is a simplified approach - you might want more sophisticated fusion
            if self.training:
                return {'base': predictions, 'refined': refined_predictions}
            else:
                return refined_predictions
        
        return predictions


class YOLOv10RefinedDetectionTrainer(YOLOv10DetectionTrainer):
    """Enhanced trainer for YOLOv10 with refinement module."""
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.training_stage = 'base'
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model with refinement capabilities."""
        model = YOLOv10RefinedDetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        
        # Setup refinement module if specified
        if hasattr(self.args, 'use_refinement') and self.args.use_refinement:
            model.setup_refinement_module()
        
        return model
    
    def setup_model(self):
        """Set up the model for training."""
        ckpt = super().setup_model()
        
        # Set training stage
        if hasattr(self.args, 'training_stage'):
            self.model.set_training_stage(self.args.training_stage)
            self.training_stage = self.args.training_stage
        
        return ckpt


class YOLOv10Refined(Model, PyTorchModelHubMixin, 
                     repo_url="https://github.com/opendatalab/DocLayout-YOLO", 
                     pipeline_tag="object-detection", 
                     license="agpl-3.0"):
    """
    Enhanced YOLOv10 model with text-based refinement module for document layout analysis.
    
    This model extends the base YOLOv10 with:
    1. OCR-based text extraction from detected regions
    2. Lightweight semantic feature extraction from text
    3. MLP-based refinement module that combines visual and text features
    4. Two-stage training approach (base model then refinement)
    """

    def __init__(self, model="yolov10m.yaml", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10RefinedDetectionModel,
                "trainer": YOLOv10RefinedDetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
    
    def train_base(self, **kwargs):
        """Train the base YOLOv10 model (Stage 1)."""
        kwargs['training_stage'] = 'base'
        kwargs['use_refinement'] = False
        return self.train(**kwargs)
    
    def train_refinement(self, base_weights=None, **kwargs):
        """Train the refinement module (Stage 2)."""
        if base_weights:
            # Load pre-trained base model
            self.load(base_weights)
        
        kwargs['training_stage'] = 'refinement'
        kwargs['use_refinement'] = True
        return self.train(**kwargs)
    
    def enable_refinement(self, enable: bool = True):
        """Enable or disable refinement module for inference."""
        if hasattr(self.model, 'enable_refinement'):
            self.model.enable_refinement(enable)
    
    def predict_with_refinement(self, source, **kwargs):
        """Run prediction with refinement module enabled."""
        self.enable_refinement(True)
        return self.predict(source, **kwargs)