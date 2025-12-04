"""
Mask Detection Module

This module provides mask detection functionality using multi-method analysis.
It analyzes facial regions to determine if a mask is present using color, texture,
edge detection, skin analysis, and brightness uniformity methods.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MaskDetector:
    """
    Mask detector using multi-method analysis.
    
    This class analyzes the mouth and nose region of a face to determine
    if a mask is present. It uses five detection methods with weighted scoring:
    - Color analysis (25%): Check for common mask colors
    - Texture analysis (20%): Masks have uniform texture
    - Edge detection (20%): Masks have distinct edges
    - Skin analysis (20%): Absence of skin tones indicates covering
    - Brightness uniformity (15%): Masks have consistent brightness
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize the MaskDetector.
        
        Args:
            threshold (float): Confidence threshold for mask detection (default: 0.5)
        """
        self.threshold = threshold
        
        # Define 8 mask color ranges in HSV color space
        # Format: (lower_bound, upper_bound)
        self.mask_colors = [
            # Blue surgical masks
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # White masks
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Black masks
            (np.array([0, 0, 0]), np.array([180, 255, 50])),
            # Surgical green
            (np.array([40, 40, 40]), np.array([80, 255, 255])),
            # Light blue
            (np.array([85, 50, 50]), np.array([100, 255, 255])),
            # Gray masks
            (np.array([0, 0, 50]), np.array([180, 30, 200])),
            # Pink/Red masks
            (np.array([160, 50, 50]), np.array([180, 255, 255])),
            # Yellow masks
            (np.array([20, 50, 50]), np.array([40, 255, 255]))
        ]
        
        # Define 5 skin tone ranges in HSV color space
        self.skin_tones = [
            # Light skin
            (np.array([0, 10, 60]), np.array([20, 150, 255])),
            # Medium skin
            (np.array([0, 20, 50]), np.array([20, 255, 255])),
            # Dark skin
            (np.array([0, 10, 0]), np.array([20, 255, 100])),
            # Asian skin tones
            (np.array([5, 30, 80]), np.array([25, 200, 255])),
            # Olive skin
            (np.array([10, 40, 60]), np.array([30, 180, 200]))
        ]
        
        # Weights for each detection method
        self.weights = {
            'color': 0.25,
            'texture': 0.20,
            'edge': 0.20,
            'skin': 0.20,
            'brightness': 0.15
        }
        
        logger.info(f"MaskDetector initialized with threshold: {threshold}")
    
    def detect_mask(self, face_roi):
        """
        Detect if a mask is present in the face region.
        
        Args:
            face_roi (numpy.ndarray): Preprocessed face region in BGR format
        
        Returns:
            tuple: (has_mask: bool, confidence: float)
                  has_mask is True if confidence > threshold
                  confidence is a value between 0.0 and 1.0
        """
        if face_roi is None or face_roi.size == 0:
            logger.warning("Invalid face_roi provided to detect_mask")
            return False, 0.0
        
        try:
            # Extract mouth and nose ROI (lower 40% of face)
            height = face_roi.shape[0]
            mouth_roi = face_roi[int(height * 0.6):, :]
            
            if mouth_roi.size == 0:
                logger.warning("Mouth ROI extraction failed")
                return False, 0.0
            
            # Run all detection methods
            color_score = self._color_analysis(mouth_roi)
            texture_score = self._texture_analysis(mouth_roi)
            edge_score = self._edge_detection(mouth_roi)
            skin_score = self._skin_analysis(mouth_roi)
            brightness_score = self._brightness_uniformity(mouth_roi)
            
            # Calculate weighted confidence score
            confidence = (
                color_score * self.weights['color'] +
                texture_score * self.weights['texture'] +
                edge_score * self.weights['edge'] +
                skin_score * self.weights['skin'] +
                brightness_score * self.weights['brightness']
            )
            
            has_mask = bool(confidence > self.threshold)
            
            logger.debug(
                f"Mask detection: confidence={confidence:.3f}, "
                f"color={color_score:.3f}, texture={texture_score:.3f}, "
                f"edge={edge_score:.3f}, skin={skin_score:.3f}, "
                f"brightness={brightness_score:.3f}"
            )
            
            return has_mask, confidence
            
        except Exception as e:
            logger.error(f"Error during mask detection: {e}")
            return False, 0.0
    
    def _color_analysis(self, mouth_roi):
        """
        Analyze colors in the mouth ROI for mask color ranges.
        
        Args:
            mouth_roi (numpy.ndarray): Mouth region in BGR format
        
        Returns:
            float: Score between 0.0 and 1.0 indicating mask color presence
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
            
            total_pixels = mouth_roi.shape[0] * mouth_roi.shape[1]
            mask_color_pixels = 0
            
            # Check each mask color range
            for lower, upper in self.mask_colors:
                mask = cv2.inRange(hsv, lower, upper)
                mask_color_pixels += np.count_nonzero(mask)
            
            # Calculate percentage of pixels matching mask colors
            score = min(mask_color_pixels / total_pixels, 1.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return 0.0
    
    def _texture_analysis(self, mouth_roi):
        """
        Analyze texture uniformity using standard deviation.
        Masks have uniform texture (low standard deviation).
        
        Args:
            mouth_roi (numpy.ndarray): Mouth region in BGR format
        
        Returns:
            float: Score between 0.0 and 1.0 indicating texture uniformity
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation
            std_dev = np.std(gray)
            
            # Lower std_dev indicates more uniform texture (mask-like)
            # Normalize: std_dev typically ranges from 0-100 for this use case
            # High uniformity (low std_dev) = high score
            score = max(0.0, 1.0 - (std_dev / 100.0))
            
            return score
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return 0.0
    
    def _edge_detection(self, mouth_roi):
        """
        Detect edges using Canny algorithm.
        Masks have distinct edges around the mouth area.
        
        Args:
            mouth_roi (numpy.ndarray): Mouth region in BGR format
        
        Returns:
            float: Score between 0.0 and 1.0 indicating edge presence
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_pixels = np.count_nonzero(edges)
            edge_density = edge_pixels / total_pixels
            
            # Masks typically have 10-30% edge density
            # Score peaks around 20% edge density
            if edge_density < 0.1:
                score = edge_density / 0.1
            elif edge_density < 0.3:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (edge_density - 0.3) / 0.3)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in edge detection: {e}")
            return 0.0
    
    def _skin_analysis(self, mouth_roi):
        """
        Analyze for skin tones in the mouth ROI.
        Absence of skin tones indicates covering (mask).
        
        Args:
            mouth_roi (numpy.ndarray): Mouth region in BGR format
        
        Returns:
            float: Score between 0.0 and 1.0 indicating absence of skin
        """
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
            
            total_pixels = mouth_roi.shape[0] * mouth_roi.shape[1]
            skin_pixels = 0
            
            # Check each skin tone range
            for lower, upper in self.skin_tones:
                mask = cv2.inRange(hsv, lower, upper)
                skin_pixels += np.count_nonzero(mask)
            
            # Calculate percentage of pixels matching skin tones
            skin_percentage = skin_pixels / total_pixels
            
            # Inverse score: less skin = higher mask probability
            score = max(0.0, 1.0 - skin_percentage)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in skin analysis: {e}")
            return 0.0
    
    def _brightness_uniformity(self, mouth_roi):
        """
        Analyze brightness uniformity across the region.
        Masks have consistent brightness.
        
        Args:
            mouth_roi (numpy.ndarray): Mouth region in BGR format
        
        Returns:
            float: Score between 0.0 and 1.0 indicating brightness uniformity
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean and standard deviation of brightness
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Lower std_brightness indicates more uniform brightness (mask-like)
            # Normalize: std_brightness typically ranges from 0-50 for this use case
            score = max(0.0, 1.0 - (std_brightness / 50.0))
            
            return score
            
        except Exception as e:
            logger.error(f"Error in brightness uniformity analysis: {e}")
            return 0.0
