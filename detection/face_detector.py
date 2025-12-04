"""
Face Detection Module

This module provides face detection functionality using OpenCV's Haar Cascade classifier.
It detects faces in video frames, extracts face regions, and preprocesses them for further analysis.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using Haar Cascade classifier.
    
    This class provides methods to:
    - Detect faces in video frames
    - Extract face regions of interest (ROI)
    - Preprocess face images for mask and helmet detection
    """
    
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        """
        Initialize the FaceDetector with Haar Cascade classifier.
        
        Args:
            cascade_path (str): Path to the Haar Cascade XML file for face detection.
                              If only filename is provided, OpenCV will search in its data directory.
        
        Raises:
            RuntimeError: If the cascade classifier cannot be loaded.
        """
        self.cascade_path = cascade_path
        
        # Try to load from OpenCV data directory first
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
        except:
            # If that fails, try the provided path directly
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Verify the cascade was loaded successfully
        if self.face_cascade.empty():
            error_msg = f"Failed to load Haar Cascade classifier from {cascade_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"FaceDetector initialized with cascade: {cascade_path}")
    
    def detect_faces(self, frame):
        """
        Detect faces in a video frame with improved filtering to reduce false positives.
        
        Args:
            frame (numpy.ndarray): BGR video frame from camera
        
        Returns:
            list: List of face rectangles as tuples (x, y, w, h)
                 Empty list if no faces detected
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to detect_faces")
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with BALANCED parameters for multiple person detection
        # Parameters:
        # - scaleFactor: 1.15 (balanced - detects faces at various distances)
        # - minNeighbors: 5 (balanced - detects both near and far faces)
        # - minSize: (40, 40) (smaller to detect faces in background)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        # Convert numpy array to list of tuples
        face_list = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        
        # Filter out overlapping faces (likely false positives like ear detected as face)
        face_list = self._filter_overlapping_faces(face_list)
        
        # Keep up to 3 largest faces to detect multiple people
        # This allows detection of people in background while filtering out very small false positives
        face_list = self._keep_largest_faces(face_list, max_faces=3)
        
        logger.debug(f"Detected {len(face_list)} face(s) in frame after filtering")
        
        return face_list
    
    def _filter_overlapping_faces(self, faces, overlap_threshold=0.3):
        """
        Filter out overlapping face detections (likely false positives).
        
        Args:
            faces: List of face rectangles [(x, y, w, h), ...]
            overlap_threshold: Minimum overlap ratio to consider faces as overlapping
        
        Returns:
            list: Filtered list of non-overlapping faces
        """
        if len(faces) <= 1:
            return faces
        
        # Sort faces by area (largest first)
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        filtered_faces = []
        for face in faces_sorted:
            # Check if this face overlaps significantly with any already accepted face
            is_overlapping = False
            for accepted_face in filtered_faces:
                if self._calculate_overlap(face, accepted_face) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def _calculate_overlap(self, face1, face2):
        """
        Calculate overlap ratio between two face rectangles.
        
        Args:
            face1: First face rectangle (x, y, w, h)
            face2: Second face rectangle (x, y, w, h)
        
        Returns:
            float: Overlap ratio (0.0 to 1.0)
        """
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Return overlap ratio relative to smaller face
        smaller_area = min(area1, area2)
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    def _keep_largest_faces(self, faces, max_faces=1):
        """
        Keep only the largest N faces (real faces are usually larger than false positives).
        Changed default to 1 to be more strict and avoid false positives from posters/photos.
        
        Args:
            faces: List of face rectangles [(x, y, w, h), ...]
            max_faces: Maximum number of faces to keep (default: 1 for single person)
        
        Returns:
            list: List of largest faces
        """
        if len(faces) <= max_faces:
            return faces
        
        # Sort by area (largest first) and keep top N
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces_sorted[:max_faces]
    
    def extract_face_roi(self, frame, face_rect):
        """
        Extract face region of interest from frame.
        
        Args:
            frame (numpy.ndarray): BGR video frame
            face_rect (tuple): Face rectangle as (x, y, w, h)
        
        Returns:
            numpy.ndarray: Extracted face region, or None if extraction fails
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to extract_face_roi")
            return None
        
        if not face_rect or len(face_rect) != 4:
            logger.warning("Invalid face_rect provided to extract_face_roi")
            return None
        
        x, y, w, h = face_rect
        
        # Validate coordinates
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            logger.warning(f"Invalid face coordinates: {face_rect}")
            return None
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        x2 = min(x + w, frame_width)
        y2 = min(y + h, frame_height)
        
        # Extract ROI
        face_roi = frame[y:y2, x:x2]
        
        if face_roi.size == 0:
            logger.warning(f"Extracted face ROI is empty for rect: {face_rect}")
            return None
        
        return face_roi
    
    def preprocess_face(self, face_roi):
        """
        Preprocess face image for mask and helmet detection.
        
        Applies the following preprocessing steps:
        1. Gaussian blur to reduce noise
        2. Convert BGR to LAB color space
        3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        4. Convert back to BGR
        
        Args:
            face_roi (numpy.ndarray): Face region in BGR format
        
        Returns:
            numpy.ndarray: Preprocessed face image in BGR format, or None if preprocessing fails
        """
        if face_roi is None or face_roi.size == 0:
            logger.warning("Invalid face_roi provided to preprocess_face")
            return None
        
        try:
            # Step 1: Apply Gaussian blur to reduce noise
            # Kernel size (5, 5) is a good balance between noise reduction and detail preservation
            blurred = cv2.GaussianBlur(face_roi, (5, 5), 0)
            
            # Step 2: Convert BGR to LAB color space
            # LAB separates luminance (L) from color information (A, B)
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            
            # Step 3: Apply CLAHE to L channel for better contrast
            # CLAHE improves local contrast and enhances details
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Step 4: Convert back to BGR
            preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            logger.debug("Face preprocessing completed successfully")
            
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error during face preprocessing: {e}")
            return None
