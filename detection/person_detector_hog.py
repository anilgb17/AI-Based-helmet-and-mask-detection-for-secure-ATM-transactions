"""
HOG Person Detector - More Reliable Multiple Person Detection
==============================================================

This module uses OpenCV's HOG (Histogram of Oriented Gradients) person detector
which is more reliable than the CNN model trained on limited data.

HOG detector is pre-trained on thousands of images and specifically designed
to detect full human bodies, not just faces.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PersonDetectorHOG:
    """
    Person detector using HOG (Histogram of Oriented Gradients).
    
    More reliable than CNN for detecting multiple people because:
    - Pre-trained on large datasets
    - Detects full body, not just face
    - Works well even when faces are not visible
    - No training data needed
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize HOG person detector.
        
        Args:
            threshold: Detection confidence threshold (0.0-1.0)
        """
        self.threshold = threshold
        
        # Initialize HOG descriptor with default people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        logger.info("HOG Person Detector initialized")
    
    def detect_multiple_persons(self, frame):
        """
        Detect if multiple persons are present in the frame.
        
        Args:
            frame: Full frame image in BGR format
        
        Returns:
            tuple: (is_multiple: bool, confidence: float, person_count: int)
        """
        if frame is None or frame.size == 0:
            return False, 0.0, 0
        
        try:
            # Resize frame for faster processing
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Detect people using HOG
            # Parameters:
            # - winStride: (8, 8) - step size for sliding window
            # - padding: (8, 8) - padding around detection window
            # - scale: 1.05 - scale factor for multi-scale detection
            (persons, weights) = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05
            )
            
            person_count = len(persons)
            
            # Calculate average confidence
            if len(weights) > 0:
                avg_confidence = float(np.mean(weights))
            else:
                avg_confidence = 0.0
            
            # Multiple persons if count >= 2
            is_multiple = person_count >= 2
            
            logger.debug(f"HOG detected {person_count} person(s), confidence: {avg_confidence:.2f}")
            
            return is_multiple, avg_confidence, person_count
            
        except Exception as e:
            logger.error(f"Error during HOG person detection: {e}")
            return False, 0.0, 0
    
    def detect_persons_boxes(self, frame):
        """
        Detect persons and return bounding boxes.
        
        Args:
            frame: Full frame image in BGR format
        
        Returns:
            list: List of person bounding boxes [(x, y, w, h), ...]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Resize for faster processing
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Detect people
            (persons, weights) = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05
            )
            
            # Scale boxes back to original size
            persons_scaled = []
            for (x, y, w, h) in persons:
                x_scaled = int(x / scale)
                y_scaled = int(y / scale)
                w_scaled = int(w / scale)
                h_scaled = int(h / scale)
                persons_scaled.append((x_scaled, y_scaled, w_scaled, h_scaled))
            
            return persons_scaled
            
        except Exception as e:
            logger.error(f"Error detecting person boxes: {e}")
            return []
