"""
Unit tests for Mask Detection Module

Tests the MaskDetector class functionality including:
- Detection with various mask types and colors
- Detection with clear faces (no mask)
- Confidence score calculations
"""

import unittest
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import detection module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.mask_detector import MaskDetector


class TestMaskDetector(unittest.TestCase):
    """Test cases for MaskDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Initialize MaskDetector with default threshold
        cls.detector = MaskDetector(threshold=0.5)
    
    @staticmethod
    def _create_face_with_blue_mask():
        """
        Create a synthetic face image with a blue surgical mask.
        
        Returns:
            numpy.ndarray: BGR face image with blue mask
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: skin tone (forehead, eyes)
        face[:120, :] = [180, 160, 140]  # Light skin tone in BGR
        
        # Lower part: blue mask (mouth and nose area)
        face[120:, :] = [200, 100, 50]  # Blue color in BGR
        
        return face
    
    @staticmethod
    def _create_face_with_white_mask():
        """
        Create a synthetic face image with a white mask.
        
        Returns:
            numpy.ndarray: BGR face image with white mask
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: skin tone
        face[:120, :] = [180, 160, 140]
        
        # Lower part: white mask
        face[120:, :] = [240, 240, 240]  # White color in BGR
        
        return face
    
    @staticmethod
    def _create_face_with_black_mask():
        """
        Create a synthetic face image with a black mask.
        
        Returns:
            numpy.ndarray: BGR face image with black mask
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: skin tone
        face[:120, :] = [180, 160, 140]
        
        # Lower part: black mask
        face[120:, :] = [20, 20, 20]  # Black color in BGR
        
        return face
    
    @staticmethod
    def _create_clear_face():
        """
        Create a synthetic face image without a mask (clear face).
        
        Returns:
            numpy.ndarray: BGR face image without mask
        """
        # Create face (200x200) with uniform skin tone
        face = np.ones((200, 200, 3), dtype=np.uint8)
        face[:, :] = [180, 160, 140]  # Light skin tone in BGR
        
        # Add some texture variation to simulate real skin
        noise = np.random.randint(-20, 20, (200, 200, 3), dtype=np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add darker region for mouth (no mask)
        face[140:180, 80:120] = [120, 100, 90]  # Darker skin for mouth area
        
        return face
    
    def test_initialization(self):
        """Test MaskDetector initialization."""
        detector = MaskDetector()
        self.assertEqual(detector.threshold, 0.5)
        self.assertEqual(len(detector.mask_colors), 8)
        self.assertEqual(len(detector.skin_tones), 5)
        
        # Test custom threshold
        detector_custom = MaskDetector(threshold=0.7)
        self.assertEqual(detector_custom.threshold, 0.7)
    
    def test_detect_blue_mask(self):
        """Test detection with blue surgical mask."""
        face = self._create_face_with_blue_mask()
        has_mask, confidence = self.detector.detect_mask(face)
        
        # Should detect mask with reasonable confidence
        self.assertIsInstance(has_mask, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Blue mask should have higher confidence
        self.assertGreater(confidence, 0.3)
    
    def test_detect_white_mask(self):
        """Test detection with white mask."""
        face = self._create_face_with_white_mask()
        has_mask, confidence = self.detector.detect_mask(face)
        
        # Should detect mask
        self.assertIsInstance(has_mask, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # White mask should have higher confidence
        self.assertGreater(confidence, 0.3)
    
    def test_detect_black_mask(self):
        """Test detection with black mask."""
        face = self._create_face_with_black_mask()
        has_mask, confidence = self.detector.detect_mask(face)
        
        # Should detect mask
        self.assertIsInstance(has_mask, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Black mask should have higher confidence
        self.assertGreater(confidence, 0.3)
    
    def test_detect_clear_face(self):
        """Test detection with clear face (no mask)."""
        face = self._create_clear_face()
        has_mask, confidence = self.detector.detect_mask(face)
        
        # Should not detect mask or have low confidence
        self.assertIsInstance(has_mask, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Clear face should have lower confidence than masked faces
        self.assertLess(confidence, 0.7)
    
    def test_confidence_score_range(self):
        """Test that confidence scores are always in valid range."""
        test_faces = [
            self._create_face_with_blue_mask(),
            self._create_face_with_white_mask(),
            self._create_face_with_black_mask(),
            self._create_clear_face()
        ]
        
        for face in test_faces:
            has_mask, confidence = self.detector.detect_mask(face)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_threshold_behavior(self):
        """Test that threshold correctly determines has_mask result."""
        face = self._create_face_with_blue_mask()
        
        # Test with low threshold
        detector_low = MaskDetector(threshold=0.1)
        has_mask_low, confidence_low = detector_low.detect_mask(face)
        
        # Test with high threshold
        detector_high = MaskDetector(threshold=0.9)
        has_mask_high, confidence_high = detector_high.detect_mask(face)
        
        # Confidence should be the same regardless of threshold
        self.assertAlmostEqual(confidence_low, confidence_high, places=5)
        
        # has_mask should depend on threshold
        if confidence_low > 0.1:
            self.assertTrue(has_mask_low)
        if confidence_high < 0.9:
            self.assertFalse(has_mask_high)
    
    def test_invalid_input(self):
        """Test detection with invalid input."""
        # Test with None
        has_mask, confidence = self.detector.detect_mask(None)
        self.assertFalse(has_mask)
        self.assertEqual(confidence, 0.0)
        
        # Test with empty array
        empty_face = np.array([])
        has_mask, confidence = self.detector.detect_mask(empty_face)
        self.assertFalse(has_mask)
        self.assertEqual(confidence, 0.0)
    
    def test_color_analysis(self):
        """Test color analysis method."""
        face = self._create_face_with_blue_mask()
        # Extract mouth ROI (lower 40%)
        height = face.shape[0]
        mouth_roi = face[int(height * 0.6):, :]
        
        score = self.detector._color_analysis(mouth_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_texture_analysis(self):
        """Test texture analysis method."""
        face = self._create_face_with_blue_mask()
        height = face.shape[0]
        mouth_roi = face[int(height * 0.6):, :]
        
        score = self.detector._texture_analysis(mouth_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_edge_detection(self):
        """Test edge detection method."""
        face = self._create_face_with_blue_mask()
        height = face.shape[0]
        mouth_roi = face[int(height * 0.6):, :]
        
        score = self.detector._edge_detection(mouth_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_skin_analysis(self):
        """Test skin analysis method."""
        face = self._create_face_with_blue_mask()
        height = face.shape[0]
        mouth_roi = face[int(height * 0.6):, :]
        
        score = self.detector._skin_analysis(mouth_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_brightness_uniformity(self):
        """Test brightness uniformity method."""
        face = self._create_face_with_blue_mask()
        height = face.shape[0]
        mouth_roi = face[int(height * 0.6):, :]
        
        score = self.detector._brightness_uniformity(mouth_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
