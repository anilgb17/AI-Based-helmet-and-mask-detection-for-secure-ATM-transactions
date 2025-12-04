"""
PyTorch-based Helmet Detection Module
Uses trained CNN model for helmet detection
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import logging
import os

logger = logging.getLogger(__name__)

class HelmetDetectorCNN(nn.Module):
    """CNN model for helmet detection"""
    def __init__(self):
        super(HelmetDetectorCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class HelmetDetectorPyTorch:
    """Helmet detector using trained PyTorch model"""
    
    def __init__(self, threshold=0.5, model_path='models/helmet_detector.pth'):
        """
        Initialize the PyTorch-based helmet detector
        
        Args:
            threshold: Confidence threshold for detection
            model_path: Path to trained model file
        """
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = HelmetDetectorCNN().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded helmet detection model from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            self.model = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_helmet(self, face_roi):
        """
        Detect if a helmet is present in the face region
        
        Args:
            face_roi: Face region image in BGR format
        
        Returns:
            tuple: (has_helmet: bool, confidence: float)
        """
        if self.model is None or face_roi is None or face_roi.size == 0:
            return False, 0.0
        
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            input_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0][0].item()  # with_helmet probability
                
            has_helmet = confidence > self.threshold
            
            return has_helmet, confidence
            
        except Exception as e:
            logger.error(f"Error during helmet detection: {e}")
            return False, 0.0
