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

        # Ensure num-classes attribute exists for refinement setup
        if not hasattr(self, 'nc') or self.nc is None:
            try:
                if nc is not None:
                    self.nc = nc
                elif hasattr(self, 'model') and hasattr(self.model, 'names'):
                    self.nc = len(self.model.names)
                else:
                    # Fallback: try detection head attribute
                    head = getattr(self, 'model', None)
                    if head is not None and hasattr(head, 'nc'):
                        self.nc = head.nc
                    else:
                        self.nc = 0
            except Exception:
                self.nc = nc if nc is not None else 0
        
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
        
    def set_training_stage(self, stage: str, freeze_backbone_layers: int = None):
        """
        Set training stage: 'base' for training base YOLO, 'refinement' for training refinement module.
        
        Args:
            stage: 'base' or 'refinement'
            freeze_backbone_layers: Number of backbone layers to freeze (None = freeze all except refinement)
        """
        if stage not in ['base', 'refinement']:
            raise ValueError("Training stage must be 'base' or 'refinement'")
        
        self.training_stage = stage
        
        if stage == 'refinement':
            if freeze_backbone_layers is not None:
                # Partial freezing strategy - freeze only first N layers
                frozen_count = 0
                for name, param in self.named_parameters():
                    if 'refinement_module' not in name and 'model' in name:
                        if frozen_count < freeze_backbone_layers:
                            param.requires_grad = False
                            frozen_count += 1
                        else:
                            param.requires_grad = True
                print(f"Partial freeze: froze first {frozen_count} backbone layers")
            else:
                # Complete freezing strategy - freeze all YOLO parameters
                for name, param in self.named_parameters():
                    if 'refinement_module' not in name:
                        param.requires_grad = False
                print("Complete freeze: froze all YOLO parameters for refinement training")
        else:
            # Unfreeze all parameters for base training
            for param in self.parameters():
                param.requires_grad = True
            print("Unfroze all parameters for base training")
    
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
    
    def forward(self, x, augment=False, profile=False, visualize=False, embed=None, **kwargs):
        """
        Forward pass with optional refinement.
        
        Args:
            x: Either a batch dict (as passed by the Trainer), or an image tensor, or a dict containing
               'images' (or Ultralytics-style 'img') plus optional 'text_features'.
            augment: Whether to use augmentation during inference
            profile: Whether to profile the model
            visualize: Whether to visualize features
            
        Returns:
            Model predictions, potentially refined with text features
        """
        # If a full batch dict is passed (trainer -> model(batch)), mimic BaseModel.forward behaviour:
        # return the computed loss by delegating to self.loss(). DetectionModel.loss will recall this
        # forward with the tensor (batch['img']) only, which then follows the normal path below.
        if isinstance(x, dict) and (('img' in x) or ('images' in x and 'cls' in x)):  # training batch
            # Delegate to loss function (will call forward again with tensor) to keep training loop intact.
            return self.loss(x)

        # Handle lightweight dict input carrying just tensors (inference with optional text features)
        text_features = None
        if isinstance(x, dict):
            # Accept either 'images' (our custom) or 'img' (ultralytics) key
            if 'images' in x:
                images = x['images']
            elif 'img' in x:
                images = x['img']
            else:
                raise KeyError("Input dict must contain 'img' or 'images' key")
            text_features = x.get('text_features', None)
        else:  # plain tensor
            images = x
        
        # Base YOLO forward pass
        if self.training_stage == 'base' or not self.use_refinement:
            return super().forward(images, augment=augment, profile=profile, visualize=visualize)

        # Refinement stage forward pass
        predictions, yolo_features = self.extract_yolo_features(images)

        if self.use_refinement and self.refinement_module is not None and text_features is not None:
            # Apply refinement
            refined_predictions = self.refinement_module(yolo_features, text_features)

            # Combine or replace predictions as needed
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
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Create dataloader with optional text feature extraction for refinement training.
        """
        # Check if we need text features (refinement stage)
        extract_text_features = (
            self.training_stage == 'refinement' and 
            hasattr(self.args, 'use_refinement') and 
            self.args.use_refinement
        )
        
        if extract_text_features:
            # Use custom refinement dataset
            from doclayout_yolo.data.refinement_dataset import create_refinement_dataloader
            
            dataloader = create_refinement_dataloader(
                dataset_path=dataset_path,
                batch_size=batch_size,
                extract_text_features=True,
                shuffle=(mode == "train"),
                num_workers=min(8, self.args.workers)
            )
            
            print(f"✓ Using refinement dataloader with text features (mode: {mode})")
            return dataloader
        else:
            # Use standard YOLO dataloader
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch with text features support.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Preprocessed batch ready for model
        """
        # Standard preprocessing
        batch = super().preprocess_batch(batch)
        
        # Preserve text features if present
        if isinstance(batch, dict) and 'text_features' in batch:
            # Move text features to device
            if batch['text_features'] is not None:
                batch['text_features'] = batch['text_features'].to(self.device, non_blocking=True)
        
        return batch
    
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
    
    def train_refinement(self, base_weights=None, freeze_backbone_layers=None, **kwargs):
        """
        Train the refinement module (Stage 2).
        
        Args:
            base_weights: Path to pre-trained base model weights
            freeze_backbone_layers: Number of backbone layers to freeze (None = freeze all)
            **kwargs: Additional training arguments
        """
        if base_weights:
            # Load pre-trained base model
            self.load(base_weights)
        
        # Setup refinement training parameters
        kwargs['training_stage'] = 'refinement'
        kwargs['use_refinement'] = True
        
        # Add freezing strategy
        if freeze_backbone_layers is not None:
            kwargs['freeze_backbone_layers'] = freeze_backbone_layers
        
        # Setup refinement module if not already done
        if hasattr(self.model, 'setup_refinement_module'):
            self.model.setup_refinement_module()
        
        return self.train(**kwargs)
    
    def enable_refinement(self, enable: bool = True):
        """Enable or disable refinement module for inference."""
        if hasattr(self.model, 'enable_refinement'):
            self.model.enable_refinement(enable)
    
    def predict_with_refinement(self, source, **kwargs):
        """Run prediction with refinement module enabled."""
        self.enable_refinement(True)
        return self.predict(source, **kwargs)