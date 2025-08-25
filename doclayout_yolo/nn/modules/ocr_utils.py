"""
OCR utilities for extracting text features from document images.
Used by the refinement module to enhance layout detection with semantic information.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
import torch
import torch.nn as nn

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class SimpleOCRExtractor:
    """Simple OCR text extractor for document layout analysis."""
    
    def __init__(self, languages=['en'], use_gpu=True):
        """
        Initialize OCR extractor.
        
        Args:
            languages: List of languages for OCR
            use_gpu: Whether to use GPU for OCR (if available)
        """
        self.languages = languages
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self._reader = None
        
        if not OCR_AVAILABLE:
            print("Warning: EasyOCR not available. Text features will be empty.")
    
    @property
    def reader(self):
        """Lazy initialization of OCR reader."""
        if self._reader is None and OCR_AVAILABLE:
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader
    
    def extract_text_from_bbox(self, image: np.ndarray, bbox: List[float]) -> str:
        """
        Extract text from a bounding box region in the image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            bbox: Bounding box coordinates [x1, y1, x2, y2] in pixel coordinates
            
        Returns:
            Extracted text string
        """
        if not OCR_AVAILABLE or self.reader is None:
            return ""
        
        try:
            # Convert bbox to integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract region
            roi = image[y1:y2, x1:x2]
            
            # Skip very small regions
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                return ""
            
            # Perform OCR
            results = self.reader.readtext(roi, detail=0)
            
            # Combine all text
            text = " ".join(results) if results else ""
            return text.strip()
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ""


class TextFeatureExtractor:
    """Extract lightweight semantic features from text."""
    
    # Keywords for different document layout classes (DocLayNet classes)
    CLASS_KEYWORDS = {
        'Caption': ['figure', 'fig', 'table', 'caption', 'source', 'note'],
        'Footnote': ['footnote', 'note', 'reference', 'citation'],
        'Formula': ['equation', 'formula', '=', '+', '-', '×', '÷', '∑', '∫'],
        'List-item': ['•', '◦', '▪', '▫', '1.', '2.', '3.', 'a)', 'b)', 'c)'],
        'Page-footer': ['page', 'footer', 'copyright', '©', 'rights', 'reserved'],
        'Page-header': ['header', 'title', 'chapter', 'section'],
        'Picture': ['image', 'photo', 'picture', 'diagram', 'illustration'],
        'Section-header': ['section', 'chapter', 'introduction', 'conclusion', 'abstract'],
        'Table': ['table', 'row', 'column', 'cell', '|', 'tab'],
        'Text': ['text', 'paragraph', 'content', 'body'],
        'Title': ['title', 'heading', 'header', 'subject']
    }
    
    def __init__(self, vocab_size=1000):
        """
        Initialize text feature extractor.
        
        Args:
            vocab_size: Maximum vocabulary size for keyword features
        """
        self.vocab_size = vocab_size
        
    def extract_features(self, text: str, bbox: List[float], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract lightweight features from text and spatial information.
        
        Args:
            text: Extracted text string
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_shape: Image shape (height, width)
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Text statistics features
        text_clean = text.lower().strip()
        
        # Basic text statistics (6 features)
        features.extend([
            len(text_clean),  # Total character count
            len(text_clean.split()),  # Word count
            sum(1 for c in text_clean if c.isdigit()) / max(1, len(text_clean)),  # Digit ratio
            sum(1 for c in text_clean if c.isupper()) / max(1, len(text_clean)),  # Uppercase ratio
            text_clean.count('.') + text_clean.count('!') + text_clean.count('?'),  # Sentence endings
            len(re.findall(r'[^\w\s]', text_clean))  # Special character count
        ])
        
        # Keyword presence features (11 features for each class)
        for class_name, keywords in self.CLASS_KEYWORDS.items():
            keyword_score = sum(1 for keyword in keywords if keyword.lower() in text_clean)
            features.append(keyword_score / max(1, len(keywords)))
        
        # Spatial features (6 features)
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            h, w = image_shape
            
            # Normalized spatial features
            features.extend([
                x1 / w,  # Left position
                y1 / h,  # Top position
                (x2 - x1) / w,  # Width ratio
                (y2 - y1) / h,  # Height ratio
                (x1 + x2) / (2 * w),  # Center X
                (y1 + y2) / (2 * h),  # Center Y
            ])
        else:
            features.extend([0.0] * 6)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        return 6 + len(self.CLASS_KEYWORDS) + 6  # text stats + class keywords + spatial


class RefinementModule(nn.Module):
    """
    Lightweight refinement module that combines YOLO features with text-based semantic features.
    """
    
    def __init__(self, yolo_feature_dim: int, text_feature_dim: int, num_classes: int, 
                 hidden_dims: List[int] = [256, 128]):
        """
        Initialize refinement module.
        
        Args:
            yolo_feature_dim: Dimension of YOLO output features
            text_feature_dim: Dimension of text features
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions for MLP
        """
        super().__init__()
        
        self.yolo_feature_dim = yolo_feature_dim
        self.text_feature_dim = text_feature_dim
        self.num_classes = num_classes
        
        # Feature fusion
        total_dim = yolo_feature_dim + text_feature_dim
        
        # MLP layers
        layers = []
        prev_dim = total_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize module weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, yolo_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of refinement module.
        
        Args:
            yolo_features: Features from YOLO backbone [batch_size, yolo_feature_dim]
            text_features: Text-based semantic features [batch_size, text_feature_dim]
            
        Returns:
            Refined class predictions [batch_size, num_classes]
        """
        # Concatenate features
        combined_features = torch.cat([yolo_features, text_features], dim=1)
        
        # Apply MLP
        output = self.mlp(combined_features)
        
        return output


def extract_text_features_batch(images: List[np.ndarray], bboxes: List[List[float]], 
                               ocr_extractor: SimpleOCRExtractor,
                               feature_extractor: TextFeatureExtractor) -> torch.Tensor:
    """
    Extract text features for a batch of images and bounding boxes.
    
    Args:
        images: List of images as numpy arrays
        bboxes: List of bounding box coordinates for each image
        ocr_extractor: OCR extractor instance
        feature_extractor: Text feature extractor instance
        
    Returns:
        Tensor of text features [batch_size, feature_dim]
    """
    batch_features = []
    
    for image, bbox_list in zip(images, bboxes):
        image_features = []
        
        for bbox in bbox_list:
            # Extract text
            text = ocr_extractor.extract_text_from_bbox(image, bbox)
            
            # Extract features
            features = feature_extractor.extract_features(text, bbox, image.shape[:2])
            image_features.append(features)
        
        if image_features:
            batch_features.append(np.stack(image_features))
        else:
            # Handle case with no detections
            empty_features = np.zeros((1, feature_extractor.get_feature_dim()), dtype=np.float32)
            batch_features.append(empty_features)
    
    # Convert to tensor
    return torch.from_numpy(np.vstack(batch_features))