"""
Enhanced dataset class for DocLayout-YOLO with OCR text feature extraction.
This dataset extends the base YOLO dataset to include text features for refinement training.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from doclayout_yolo.data.dataset import YOLODataset
from doclayout_yolo.nn.modules.ocr_utils import SimpleOCRExtractor, TextFeatureExtractor


class RefinementDataset(YOLODataset):
    """
    Dataset class that adds OCR-based text feature extraction to YOLO training.
    
    This dataset extracts text features from ground truth bounding boxes during training,
    which are then used by the refinement module to improve predictions.
    """
    
    def __init__(self, *args, extract_text_features=False, **kwargs):
        """
        Initialize the refinement dataset.
        
        Args:
            extract_text_features: Whether to extract text features from images
            *args, **kwargs: Arguments passed to parent YOLODataset
        """
        super().__init__(*args, **kwargs)
        
        self.extract_text_features = extract_text_features
        
        if self.extract_text_features:
            self.ocr_extractor = SimpleOCRExtractor()
            self.text_feature_extractor = TextFeatureExtractor()
            print(f"✓ OCR text feature extraction enabled (dim: {self.text_feature_extractor.get_feature_dim()})")
        else:
            self.ocr_extractor = None
            self.text_feature_extractor = None
    
    def __getitem__(self, index):
        """
        Get dataset item with optional text features.
        
        Returns:
            Dictionary containing image, labels, and optionally text features
        """
        # Get base dataset item (image, labels, etc.)
        item = super().__getitem__(index)
        
        # If text feature extraction is disabled, return base item
        if not self.extract_text_features or self.ocr_extractor is None:
            return item
        
        # Extract text features from ground truth bounding boxes
        try:
            text_features = self._extract_text_features_from_item(item)
            if text_features is not None:
                item['text_features'] = text_features
        except Exception as e:
            # If text extraction fails, continue without text features
            print(f"Warning: Text extraction failed for index {index}: {e}")
        
        return item
    
    def _extract_text_features_from_item(self, item: Dict) -> Optional[torch.Tensor]:
        """
        Extract text features from a dataset item using ground truth bounding boxes.
        
        Args:
            item: Dataset item containing 'im_file', 'img', 'labels', etc.
            
        Returns:
            Text features tensor or None if extraction fails
        """
        # Get image and labels
        img = item.get('img')  # Already processed image (HWC format)
        labels = item.get('labels')  # Ground truth labels
        
        if img is None or labels is None or len(labels) == 0:
            return None
        
        # Convert image tensor to numpy for OCR
        if isinstance(img, torch.Tensor):
            # Convert from CHW to HWC and to numpy
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.dtype == np.float32:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img
        
        # Get image dimensions
        h, w = img_np.shape[:2]
        
        # Extract text features for each bounding box
        text_features_list = []
        
        for label in labels:
            # Parse label (class_id, x_center, y_center, width, height) in normalized format
            if len(label) >= 5:
                class_id, x_center, y_center, box_width, box_height = label[:5]
                
                # Convert normalized coordinates to pixel coordinates
                x1 = max(0, int((x_center - box_width / 2) * w))
                y1 = max(0, int((y_center - box_height / 2) * h))
                x2 = min(w, int((x_center + box_width / 2) * w))
                y2 = min(h, int((y_center + box_height / 2) * h))
                
                bbox = [x1, y1, x2, y2]
                
                # Extract text from bounding box
                try:
                    text = self.ocr_extractor.extract_text_from_bbox(img_np, bbox)
                    
                    # Extract features from text
                    features = self.text_feature_extractor.extract_features(
                        text, bbox, (h, w)
                    )
                    text_features_list.append(features)
                    
                except Exception as e:
                    # If OCR fails for this box, use zero features
                    zero_features = np.zeros(self.text_feature_extractor.get_feature_dim(), dtype=np.float32)
                    text_features_list.append(zero_features)
        
        if text_features_list:
            # Stack features and convert to tensor
            text_features = np.stack(text_features_list)
            return torch.from_numpy(text_features)
        
        return None
    
    @staticmethod
    def collate_fn_with_text_features(batch):
        """
        Custom collate function that handles text features.
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched data including text features
        """
        # Separate items with and without text features
        images = []
        labels = []
        text_features = []
        
        for item in batch:
            images.append(item['img'])
            if 'labels' in item:
                labels.append(item['labels'])
            if 'text_features' in item:
                text_features.append(item['text_features'])
        
        # Stack images
        images = torch.stack(images)
        
        # Create batched output
        batch_dict = {
            'img': images,
            'labels': labels if labels else None
        }
        
        # Add text features if available
        if text_features:
            # Pad text features to same length within batch
            max_boxes = max(tf.shape[0] for tf in text_features)
            feature_dim = text_features[0].shape[1]
            
            padded_text_features = []
            for tf in text_features:
                if tf.shape[0] < max_boxes:
                    # Pad with zeros
                    padding = torch.zeros(max_boxes - tf.shape[0], feature_dim)
                    tf_padded = torch.cat([tf, padding], dim=0)
                else:
                    tf_padded = tf[:max_boxes]  # Truncate if too long
                padded_text_features.append(tf_padded)
            
            batch_dict['text_features'] = torch.stack(padded_text_features)
        
        return batch_dict


def create_refinement_dataloader(dataset_path: str, batch_size: int = 8, 
                               extract_text_features: bool = True,
                               **kwargs) -> torch.utils.data.DataLoader:
    """
    Create a dataloader with text feature extraction for refinement training.
    
    Args:
        dataset_path: Path to dataset yaml file
        batch_size: Batch size for training
        extract_text_features: Whether to extract text features
        **kwargs: Additional arguments for dataset/dataloader
        
    Returns:
        DataLoader with text feature extraction
    """
    # Create dataset with text feature extraction
    dataset = RefinementDataset(
        path=dataset_path,
        extract_text_features=extract_text_features,
        **kwargs
    )
    
    # Create dataloader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get('shuffle', True),
        num_workers=kwargs.get('num_workers', 4),
        collate_fn=RefinementDataset.collate_fn_with_text_features,
        pin_memory=kwargs.get('pin_memory', True)
    )
    
    return dataloader