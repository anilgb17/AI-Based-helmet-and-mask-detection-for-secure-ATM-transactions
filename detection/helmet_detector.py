"""
Helmet Detection Module

Analyzes head regions to determine if a helmet or head covering is present
using multi-method analysis including color, texture, brightness, and hair detection.
"""

import cv2
import numpy as np


class HelmetDetector:
    """
    Detects helmets on faces using multiple analysis methods.
    
    Detection Methods:
    1. Color Analysis (Weight: 0.30): Check for helmet colors
    2. Texture Uniformity (Weight: 0.25): Helmets have smooth, uniform texture
    3. Brightness Check (Weight: 0.25): Helmets have reflective or consistent brightness
    4. Hair Detection (Weight: 0.20): Absence of hair texture indicates covering
    
    Threshold: Confidence > 0.4 indicates helmet detected
    ROI: Upper 30% of face (forehead and top head region)
    """
    
    def __init__(self, threshold=0.4):
        """
        Initialize HelmetDetector with detection threshold.
        
        Args:
            threshold (float): Confidence threshold for helmet detection (default: 0.4)
        """
        self.threshold = threshold
        
        # Define helmet color ranges in HSV
        # Format: (lower_bound, upper_bound)
        self.helmet_colors = [
            # Red helmets (two ranges due to HSV wrap-around)
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255])),
            # Yellow helmets
            (np.array([20, 50, 50]), np.array([30, 255, 255])),
            # White helmets
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Black helmets
            (np.array([0, 0, 0]), np.array([180, 255, 50])),
            # Blue helmets
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
        ]
        
        # Weights for each detection method
        self.weights = {
            'color': 0.30,
            'texture': 0.25,
            'brightness': 0.25,
            'hair': 0.20
        }
    
    def detect_helmet(self, face_roi):
        """
        Main method to detect helmet on a face.
        
        Args:
            face_roi (numpy.ndarray): Face region image in BGR format
        
        Returns:
            tuple: (has_helmet: bool, confidence: float)
        """
        # Validate input
        if face_roi is None or face_roi.size == 0:
            return False, 0.0
        
        try:
            # Extract head ROI (upper 30% of face)
            height = face_roi.shape[0]
            head_roi = face_roi[:int(height * 0.3), :]
            
            if head_roi.size == 0:
                return False, 0.0
            
            # Apply all detection methods
            color_score = self._color_analysis(head_roi)
            texture_score = self._texture_uniformity(head_roi)
            brightness_score = self._brightness_check(head_roi)
            hair_score = self._hair_detection(head_roi)
            
            # Calculate weighted confidence score
            confidence = (
                color_score * self.weights['color'] +
                texture_score * self.weights['texture'] +
                brightness_score * self.weights['brightness'] +
                hair_score * self.weights['hair']
            )
            
            # Ensure confidence is in valid range
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Determine if helmet is detected based on threshold
            has_helmet = bool(confidence > self.threshold)
            
            return has_helmet, float(confidence)
            
        except Exception as e:
            # Return safe defaults on error
            return False, 0.0
    
    def _color_analysis(self, head_roi):
        """
        Analyze head ROI for helmet colors.
        
        Args:
            head_roi (numpy.ndarray): Head region image in BGR format
        
        Returns:
            float: Color analysis score (0.0 to 1.0)
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            
            # Check each helmet color range
            total_pixels = head_roi.shape[0] * head_roi.shape[1]
            max_match_ratio = 0.0
            
            for lower, upper in self.helmet_colors:
                # Create mask for this color range
                mask = cv2.inRange(hsv, lower, upper)
                match_pixels = cv2.countNonZero(mask)
                match_ratio = match_pixels / total_pixels
                
                # Track the highest match ratio
                max_match_ratio = max(max_match_ratio, match_ratio)
            
            # Normalize score (higher ratio = more likely helmet)
            score = min(max_match_ratio * 2.0, 1.0)
            
            return score
            
        except Exception:
            return 0.0
    
    def _texture_uniformity(self, head_roi):
        """
        Analyze texture uniformity (helmets have smooth, uniform texture).
        
        Args:
            head_roi (numpy.ndarray): Head region image in BGR format
        
        Returns:
            float: Texture uniformity score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation (lower = more uniform)
            std_dev = np.std(gray)
            
            # Normalize: low std_dev (< 20) indicates uniform texture (helmet)
            # High std_dev (> 40) indicates varied texture (hair)
            if std_dev < 20:
                score = 1.0
            elif std_dev > 40:
                score = 0.0
            else:
                # Linear interpolation between 20 and 40
                score = 1.0 - ((std_dev - 20) / 20)
            
            return score
            
        except Exception:
            return 0.0
    
    def _brightness_check(self, head_roi):
        """
        Analyze brightness levels (helmets often have reflective or consistent brightness).
        
        Args:
            head_roi (numpy.ndarray): Head region image in BGR format
        
        Returns:
            float: Brightness score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean brightness
            mean_brightness = np.mean(gray)
            
            # Calculate brightness variance
            brightness_variance = np.var(gray)
            
            # Helmets tend to have either high brightness (reflective) or low variance (uniform)
            # Score based on both factors
            
            # High brightness score (> 150 is bright/reflective)
            brightness_score = min(mean_brightness / 150, 1.0) if mean_brightness > 100 else 0.0
            
            # Low variance score (< 200 is uniform)
            variance_score = 1.0 - min(brightness_variance / 200, 1.0)
            
            # Combine both scores
            score = max(brightness_score, variance_score)
            
            return score
            
        except Exception:
            return 0.0
    
    def _hair_detection(self, head_roi):
        """
        Inverse hair detection (absence of hair indicates helmet).
        
        Args:
            head_roi (numpy.ndarray): Head region image in BGR format
        
        Returns:
            float: Hair detection score (0.0 to 1.0, higher = less hair = more likely helmet)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection to find hair texture
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels (hair has many fine edges)
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = head_roi.shape[0] * head_roi.shape[1]
            edge_ratio = edge_pixels / total_pixels
            
            # Inverse score: low edge ratio = less hair = more likely helmet
            # Hair typically has edge ratio > 0.15
            # Smooth helmet has edge ratio < 0.05
            if edge_ratio < 0.05:
                score = 1.0
            elif edge_ratio > 0.15:
                score = 0.0
            else:
                # Linear interpolation between 0.05 and 0.15
                score = 1.0 - ((edge_ratio - 0.05) / 0.10)
            
            return score
            
        except Exception:
            return 0.0
