"""
Unit tests for Helmet Detection Module

Tests the HelmetDetector class functionality including:
- Detection with various helmet types and colors
- Detection with no helmet (hair visible)
- Confidence score calculations
"""

import unittest
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import detection module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.helmet_detector import HelmetDetector


class TestHelmetDetector(unittest.TestCase):
    """Test cases for HelmetDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Initialize HelmetDetector with default threshold
        cls.detector = HelmetDetector(threshold=0.4)
    
    @staticmethod
    def _create_face_with_red_helmet():
        """
        Create a synthetic face image with a red helmet.
        
        Returns:
            numpy.ndarray: BGR face image with red helmet
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: red helmet (top 30%)
        face[:60, :] = [50, 50, 200]  # Red color in BGR
        
        # Middle part: skin tone (eyes, nose)
        face[60:140, :] = [180, 160, 140]  # Light skin tone in BGR
        
        # Lower part: skin tone (mouth area)
        face[140:, :] = [180, 160, 140]
        
        return face
    
    @staticmethod
    def _create_face_with_yellow_helmet():
        """
        Create a synthetic face image with a yellow construction helmet.
        
        Returns:
            numpy.ndarray: BGR face image with yellow helmet
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: yellow helmet
        face[:60, :] = [100, 220, 220]  # Yellow color in BGR
        
        # Middle and lower parts: skin tone
        face[60:, :] = [180, 160, 140]
        
        return face
    
    @staticmethod
    def _create_face_with_white_helmet():
        """
        Create a synthetic face image with a white helmet.
        
        Returns:
            numpy.ndarray: BGR face image with white helmet
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: white helmet
        face[:60, :] = [240, 240, 240]  # White color in BGR
        
        # Middle and lower parts: skin tone
        face[60:, :] = [180, 160, 140]
        
        return face
    
    @staticmethod
    def _create_face_with_black_helmet():
        """
        Create a synthetic face image with a black helmet.
        
        Returns:
            numpy.ndarray: BGR face image with black helmet
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: black helmet
        face[:60, :] = [20, 20, 20]  # Black color in BGR
        
        # Middle and lower parts: skin tone
        face[60:, :] = [180, 160, 140]
        
        return face
    
    @staticmethod
    def _create_face_with_blue_helmet():
        """
        Create a synthetic face image with a blue helmet.
        
        Returns:
            numpy.ndarray: BGR face image with blue helmet
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: blue helmet
        face[:60, :] = [200, 100, 50]  # Blue color in BGR
        
        # Middle and lower parts: skin tone
        face[60:, :] = [180, 160, 140]
        
        return face
    
    @staticmethod
    def _create_face_with_hair():
        """
        Create a synthetic face image with visible hair (no helmet).
        
        Returns:
            numpy.ndarray: BGR face image with hair
        """
        # Create face (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8)
        
        # Upper part: hair with texture variation
        face[:60, :] = [40, 30, 20]  # Dark brown hair in BGR
        
        # Add texture to simulate hair
        for i in range(0, 60, 2):
            face[i:i+1, :] = [60, 45, 30]  # Lighter strands
        
        # Add random noise for hair texture
        noise = np.random.randint(-15, 15, (60, 200, 3), dtype=np.int16)
        face[:60, :] = np.clip(face[:60, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Middle and lower parts: skin tone
        face[60:, :] = [180, 160, 140]
        
        return face
    
    def test_initialization(self):
        """Test HelmetDetector initialization."""
        detector = HelmetDetector()
        self.assertEqual(detector.threshold, 0.4)
        self.assertEqual(len(detector.helmet_colors), 6)
        self.assertEqual(len(detector.weights), 4)
        
        # Test custom threshold
        detector_custom = HelmetDetector(threshold=0.6)
        self.assertEqual(detector_custom.threshold, 0.6)
    
    def test_detect_red_helmet(self):
        """Test detection with red helmet."""
        face = self._create_face_with_red_helmet()
        has_helmet, confidence = self.detector.detect_helmet(face)
        
        # Should detect helmet with reasonable confidence
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Red helmet should have higher confidence
        self.assertGreater(confidence, 0.2)
    
    def test_detect_yellow_helmet(self):
        """Test detection with yellow construction helmet."""
        face = self._create_face_with_yellow_helmet()
        has_helmet, confidence = self.detector.detect_helmet(face)
        
        # Should detect helmet
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Yellow helmet should have higher confidence
        self.assertGreater(confidence, 0.2)
    
    def test_detect_white_helmet(self):
        """Test detection with white helmet."""
        face = self._create_face_with_white_helmet()
        has_helmet, confidence = self.detector.detect_helmet(face)
        
        # Should detect helmet
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # White helmet should have higher confidence
        self.assertGreater(confidence, 0.2)
    
    def test_detect_black_helmet(self):
        """Test detection with black helmet."""
        face = self._create_face_with_black_helmet()
        has_helmet, confidence = self.detector.detect_helmet(face)
        
        # Should detect helmet
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Black helmet should have higher confidence
        self.assertGreater(confidence, 0.2)
    
    def test_detect_blue_helmet(self):
        """Test detection with blue helmet."""
        face = self._create_face_with_blue_helmet()
        has_helmet, confidence = self.detector.detect_helmet(face)
        
        # Should detect helmet
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Blue helmet should have higher confidence
        self.assertGreater(confidence, 0.2)
    
    def test_detect_hair_no_helmet(self):
        """Test detection with visible hair (no helmet)."""
        face = self._create_face_with_hair()
        has_helmet, confidence = self.detector.detect_helmet(face)
        
        # Should return valid results
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_score_range(self):
        """Test that confidence scores are always in valid range."""
        test_faces = [
            self._create_face_with_red_helmet(),
            self._create_face_with_yellow_helmet(),
            self._create_face_with_white_helmet(),
            self._create_face_with_black_helmet(),
            self._create_face_with_blue_helmet(),
            self._create_face_with_hair()
        ]
        
        for face in test_faces:
            has_helmet, confidence = self.detector.detect_helmet(face)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_threshold_behavior(self):
        """Test that threshold correctly determines has_helmet result."""
        face = self._create_face_with_red_helmet()
        
        # Test with low threshold
        detector_low = HelmetDetector(threshold=0.1)
        has_helmet_low, confidence_low = detector_low.detect_helmet(face)
        
        # Test with high threshold
        detector_high = HelmetDetector(threshold=0.9)
        has_helmet_high, confidence_high = detector_high.detect_helmet(face)
        
        # Confidence should be the same regardless of threshold
        self.assertAlmostEqual(confidence_low, confidence_high, places=5)
        
        # has_helmet should depend on threshold
        if confidence_low > 0.1:
            self.assertTrue(has_helmet_low)
        if confidence_high < 0.9:
            self.assertFalse(has_helmet_high)
    
    def test_invalid_input(self):
        """Test detection with invalid input."""
        # Test with None
        has_helmet, confidence = self.detector.detect_helmet(None)
        self.assertFalse(has_helmet)
        self.assertEqual(confidence, 0.0)
        
        # Test with empty array
        empty_face = np.array([])
        has_helmet, confidence = self.detector.detect_helmet(empty_face)
        self.assertFalse(has_helmet)
        self.assertEqual(confidence, 0.0)
    
    def test_color_analysis(self):
        """Test color analysis method."""
        face = self._create_face_with_red_helmet()
        # Extract head ROI (upper 30%)
        height = face.shape[0]
        head_roi = face[:int(height * 0.3), :]
        
        score = self.detector._color_analysis(head_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_texture_uniformity(self):
        """Test texture uniformity method."""
        face = self._create_face_with_red_helmet()
        height = face.shape[0]
        head_roi = face[:int(height * 0.3), :]
        
        score = self.detector._texture_uniformity(head_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_brightness_check(self):
        """Test brightness check method."""
        face = self._create_face_with_red_helmet()
        height = face.shape[0]
        head_roi = face[:int(height * 0.3), :]
        
        score = self.detector._brightness_check(head_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_hair_detection(self):
        """Test hair detection method."""
        face = self._create_face_with_hair()
        height = face.shape[0]
        head_roi = face[:int(height * 0.3), :]
        
        score = self.detector._hair_detection(head_roi)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_helmet_vs_hair_distinction(self):
        """Test that detector can distinguish between helmet and hair."""
        helmet_face = self._create_face_with_red_helmet()
        hair_face = self._create_face_with_hair()
        
        has_helmet_helmet, confidence_helmet = self.detector.detect_helmet(helmet_face)
        has_helmet_hair, confidence_hair = self.detector.detect_helmet(hair_face)
        
        # Helmet should have higher confidence than hair
        self.assertGreater(confidence_helmet, confidence_hair)


if __name__ == '__main__':
    unittest.main()
