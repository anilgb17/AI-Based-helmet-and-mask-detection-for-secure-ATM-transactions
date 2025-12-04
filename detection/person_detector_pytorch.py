"""
PyTorch-based Person Detection Module
Uses trained CNN model to detect single vs multiple persons
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import logging
import os

logger = logging.getLogger(__name__)

class PersonDetectorCNN(nn.Module):
    """CNN model for person detection"""
    def __init__(self):
        super(PersonDetectorCNN, self).__init__()
        
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

class PersonDetectorPyTorch:
    """Person detector using trained PyTorch model"""
    
    def __init__(self, threshold=0.5, model_path='models/person_detector.pth'):
        """
        Initialize the PyTorch-based person detector
        
        Args:
            threshold: Confidence threshold for detection
            model_path: Path to trained model file
        """
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = PersonDetectorCNN().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded person detection model from {model_path}")
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
    
    def detect_multiple_persons(self, frame):
        """
        Detect if multiple persons are present in the frame
        
        Args:
            frame: Full frame image in BGR format
        
        Returns:
            tuple: (is_multiple: bool, confidence: float)
        """
        if self.model is None or frame is None or frame.size == 0:
            return False, 0.0
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                # Index 0: multiple_person, Index 1: single_person
                multiple_confidence = probabilities[0][0].item()
                
            is_multiple = multiple_confidence > self.threshold
            
            return is_multiple, multiple_confidence
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return False, 0.0
