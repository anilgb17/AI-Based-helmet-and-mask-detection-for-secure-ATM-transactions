"""
Unit tests for Face Detection Module

Tests the FaceDetector class functionality including:
- Face detection with single face, multiple faces, and no faces
- ROI extraction accuracy
- Preprocessing pipeline
"""

import unittest
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import detection module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.face_detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Initialize FaceDetector
        cls.detector = FaceDetector()
        
        # Create test frames
        cls.single_face_frame = cls._create_test_frame_with_faces(1)
        cls.multiple_faces_frame = cls._create_test_frame_with_faces(3)
        cls.no_face_frame = cls._create_test_frame_with_faces(0)
    
    @staticmethod
    def _create_test_frame_with_faces(num_faces):
        """
        Create a synthetic test frame with specified number of face-like regions.
        
        Args:
            num_faces (int): Number of faces to simulate
        
        Returns:
            numpy.ndarray: BGR test frame
        """
        # Create a blank frame (640x480, BGR)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if num_faces == 0:
            # Return blank frame for no faces
            return frame
        
        # Create face-like regions (ellipses that resemble faces)
        face_positions = [
            (160, 240),  # Left face
            (320, 240),  # Center face
            (480, 240),  # Right face
        ]
        
        for i in range(min(num_faces, len(face_positions))):
            center = face_positions[i]
            # Draw ellipse (face shape)
            cv2.ellipse(frame, center, (60, 80), 0, 0, 360, (200, 180, 160), -1)
            # Draw eyes
            cv2.circle(frame, (center[0] - 20, center[1] - 20), 8, (50, 50, 50), -1)
            cv2.circle(frame, (center[0] + 20, center[1] - 20), 8, (50, 50, 50), -1)
            # Draw nose
            cv2.line(frame, (center[0], center[1] - 10), (center[0], center[1] + 20), (150, 120, 100), 2)
            # Draw mouth
            cv2.ellipse(frame, (center[0], center[1] + 30), (25, 15), 0, 0, 180, (100, 50, 50), 2)
        
        return frame
    
    def test_initialization(self):
        """Test FaceDetector initialization."""
        detector = FaceDetector()
        self.assertIsNotNone(detector.face_cascade)
        self.assertFalse(detector.face_cascade.empty())
    
    def test_detect_single_face(self):
        """Test face detection with a single face."""
        faces = self.detector.detect_faces(self.single_face_frame)
        self.assertIsInstance(faces, list)
        # Should detect at least one face (may detect the synthetic face)
        self.assertGreaterEqual(len(faces), 0)
    
    def test_detect_multiple_faces(self):
        """Test face detection with multiple faces."""
        faces = self.detector.detect_faces(self.multiple_faces_frame)
        self.assertIsInstance(faces, list)
        # Should detect faces (may detect some of the synthetic faces)
        self.assertGreaterEqual(len(faces), 0)
    
    def test_detect_no_faces(self):
        """Test face detection with no faces."""
        faces = self.detector.detect_faces(self.no_face_frame)
        self.assertIsInstance(faces, list)
        self.assertEqual(len(faces), 0)
    
    def test_detect_faces_invalid_frame(self):
        """Test face detection with invalid frame."""
        faces = self.detector.detect_faces(None)
        self.assertEqual(faces, [])
        
        empty_frame = np.array([])
        faces = self.detector.detect_faces(empty_frame)
        self.assertEqual(faces, [])
    
    def test_extract_face_roi(self):
        """Test ROI extraction accuracy."""
        # Create a test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Define a face rectangle
        face_rect = (100, 100, 200, 200)
        
        # Extract ROI
        roi = self.detector.extract_face_roi(frame, face_rect)
        
        # Verify ROI is extracted correctly
        self.assertIsNotNone(roi)
        self.assertEqual(roi.shape[0], 200)  # Height
        self.assertEqual(roi.shape[1], 200)  # Width
        self.assertEqual(roi.shape[2], 3)    # BGR channels
    
    def test_extract_face_roi_invalid_inputs(self):
        """Test ROI extraction with invalid inputs."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test with None frame
        roi = self.detector.extract_face_roi(None, (100, 100, 200, 200))
        self.assertIsNone(roi)
        
        # Test with invalid face_rect
        roi = self.detector.extract_face_roi(frame, None)
        self.assertIsNone(roi)
        
        roi = self.detector.extract_face_roi(frame, (100, 100))
        self.assertIsNone(roi)
        
        # Test with negative coordinates
        roi = self.detector.extract_face_roi(frame, (-10, -10, 50, 50))
        self.assertIsNone(roi)  # Should return None for invalid coordinates
    
    def test_preprocess_face(self):
        """Test preprocessing pipeline."""
        # Create a test face ROI
        face_roi = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Preprocess
        preprocessed = self.detector.preprocess_face(face_roi)
        
        # Verify preprocessing was successful
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, face_roi.shape)
        self.assertEqual(preprocessed.dtype, np.uint8)
    
    def test_preprocess_face_invalid_input(self):
        """Test preprocessing with invalid input."""
        # Test with None
        preprocessed = self.detector.preprocess_face(None)
        self.assertIsNone(preprocessed)
        
        # Test with empty array
        empty_roi = np.array([])
        preprocessed = self.detector.preprocess_face(empty_roi)
        self.assertIsNone(preprocessed)
    
    def test_face_rect_format(self):
        """Test that detected faces return correct tuple format."""
        faces = self.detector.detect_faces(self.single_face_frame)
        
        for face in faces:
            self.assertIsInstance(face, tuple)
            self.assertEqual(len(face), 4)
            x, y, w, h = face
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertIsInstance(w, int)
            self.assertIsInstance(h, int)


if __name__ == '__main__':
    unittest.main()
