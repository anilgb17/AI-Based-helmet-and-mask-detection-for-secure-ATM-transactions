"""
Unit tests for Video Stream Handler Module

Tests the VideoStreamHandler class functionality including:
- Frame generation
- Overlay drawing
- Status text rendering
"""

import unittest
import cv2
import numpy as np
import sys
import os
from unittest.mock import Mock, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web.video_stream_handler import VideoStreamHandler


class TestVideoStreamHandler(unittest.TestCase):
    """Test cases for VideoStreamHandler class."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.handler = VideoStreamHandler()
        
        # Create a test frame
        self.test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Create mock security status
        self.security_status_granted = {
            'access_granted': True,
            'violation_type': None,
            'face_count': 1,
            'has_mask': False,
            'has_helmet': False,
            'mask_confidence': 0.1,
            'helmet_confidence': 0.05
        }
        
        self.security_status_denied = {
            'access_granted': False,
            'violation_type': 'mask_detected',
            'face_count': 1,
            'has_mask': True,
            'has_helmet': False,
            'mask_confidence': 0.75,
            'helmet_confidence': 0.1
        }
    
    def test_initialization(self):
        """Test VideoStreamHandler initialization."""
        handler = VideoStreamHandler()
        self.assertIsNotNone(handler)
    
    def test_draw_face_boxes_access_granted(self):
        """Test drawing face boxes with access granted (green)."""
        frame = self.test_frame.copy()
        faces = [(100, 100, 200, 200)]
        
        self.handler._draw_face_boxes(frame, faces, self.security_status_granted)
        
        # Verify frame was modified (not equal to original)
        self.assertFalse(np.array_equal(frame, self.test_frame))
        
        # Check that green color is present in the frame (access granted)
        # Green is (0, 255, 0) in BGR
        green_pixels = np.sum((frame[:, :, 1] == 255) & (frame[:, :, 0] == 0) & (frame[:, :, 2] == 0))
        self.assertGreater(green_pixels, 0, "Green color should be present for access granted")
    
    def test_draw_face_boxes_access_denied(self):
        """Test drawing face boxes with access denied (red)."""
        frame = self.test_frame.copy()
        faces = [(100, 100, 200, 200)]
        
        self.handler._draw_face_boxes(frame, faces, self.security_status_denied)
        
        # Verify frame was modified
        self.assertFalse(np.array_equal(frame, self.test_frame))
        
        # Check that red color is present in the frame (access denied)
        # Red is (0, 0, 255) in BGR
        red_pixels = np.sum((frame[:, :, 2] == 255) & (frame[:, :, 0] == 0) & (frame[:, :, 1] == 0))
        self.assertGreater(red_pixels, 0, "Red color should be present for access denied")
    
    def test_draw_face_boxes_multiple_faces(self):
        """Test drawing boxes for multiple faces."""
        frame = self.test_frame.copy()
        faces = [(50, 50, 100, 100), (300, 200, 120, 120), (500, 350, 80, 80)]
        
        self.handler._draw_face_boxes(frame, faces, self.security_status_granted)
        
        # Verify frame was modified
        self.assertFalse(np.array_equal(frame, self.test_frame))
    
    def test_draw_face_boxes_no_faces(self):
        """Test drawing with no faces detected."""
        frame = self.test_frame.copy()
        faces = []
        
        # Should not raise an error
        self.handler._draw_face_boxes(frame, faces, self.security_status_granted)
    
    def test_draw_face_boxes_with_mask_label(self):
        """Test drawing face boxes with mask detection label."""
        frame = self.test_frame.copy()
        faces = [(100, 100, 200, 200)]
        
        self.handler._draw_face_boxes(frame, faces, self.security_status_denied)
        
        # Verify frame was modified (labels should be drawn)
        self.assertFalse(np.array_equal(frame, self.test_frame))
    
    def test_draw_status_text_access_granted(self):
        """Test drawing status text for access granted."""
        frame = self.test_frame.copy()
        
        self.handler._draw_status_text(frame, self.security_status_granted)
        
        # Verify frame was modified
        self.assertFalse(np.array_equal(frame, self.test_frame))
        
        # Check that green color is present (for "ACCESS GRANTED" text)
        green_pixels = np.sum((frame[:, :, 1] == 255) & (frame[:, :, 0] == 0) & (frame[:, :, 2] == 0))
        self.assertGreater(green_pixels, 0, "Green text should be present for access granted")
    
    def test_draw_status_text_access_denied(self):
        """Test drawing status text for access denied."""
        frame = self.test_frame.copy()
        
        self.handler._draw_status_text(frame, self.security_status_denied)
        
        # Verify frame was modified
        self.assertFalse(np.array_equal(frame, self.test_frame))
        
        # Check that red color is present (for "ACCESS DENIED" text)
        red_pixels = np.sum((frame[:, :, 2] == 255) & (frame[:, :, 0] == 0) & (frame[:, :, 1] == 0))
        self.assertGreater(red_pixels, 0, "Red text should be present for access denied")
    
    def test_draw_overlays(self):
        """Test drawing complete overlays (face boxes + status text)."""
        frame = self.test_frame.copy()
        faces = [(100, 100, 200, 200)]
        
        self.handler._draw_overlays(frame, faces, self.security_status_granted)
        
        # Verify frame was modified
        self.assertFalse(np.array_equal(frame, self.test_frame))
    
    def test_generate_frames_basic(self):
        """Test frame generation with mocked components."""
        # Create mock camera
        mock_camera = Mock()
        mock_camera.read.side_effect = [
            (True, self.test_frame.copy()),
            (True, self.test_frame.copy()),
            (False, None)  # Stop after 2 frames
        ]
        
        # Create mock detectors
        mock_face_detector = Mock()
        mock_face_detector.detect_faces.return_value = [(100, 100, 200, 200)]
        mock_face_detector.extract_face_roi.return_value = np.ones((200, 200, 3), dtype=np.uint8) * 128
        mock_face_detector.preprocess_face.return_value = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        mock_mask_detector = Mock()
        mock_mask_detector.detect_mask.return_value = (False, 0.1)
        
        mock_helmet_detector = Mock()
        mock_helmet_detector.detect_helmet.return_value = (False, 0.05)
        
        mock_security_manager = Mock()
        mock_security_manager.get_status.return_value = self.security_status_granted
        
        # Generate frames
        frame_count = 0
        for frame_data in self.handler.generate_frames(
            mock_camera,
            mock_face_detector,
            mock_mask_detector,
            mock_helmet_detector,
            mock_security_manager
        ):
            # Verify frame data format
            self.assertIsInstance(frame_data, bytes)
            self.assertTrue(frame_data.startswith(b'--frame\r\n'))
            self.assertIn(b'Content-Type: image/jpeg', frame_data)
            frame_count += 1
        
        # Should have generated 2 frames before camera.read returned False
        self.assertEqual(frame_count, 2)
        
        # Verify detectors were called
        self.assertEqual(mock_face_detector.detect_faces.call_count, 2)
        self.assertEqual(mock_security_manager.update_status.call_count, 2)
    
    def test_generate_frames_no_faces(self):
        """Test frame generation when no faces are detected."""
        # Create mock camera
        mock_camera = Mock()
        mock_camera.read.side_effect = [
            (True, self.test_frame.copy()),
            (False, None)
        ]
        
        # Create mock detectors
        mock_face_detector = Mock()
        mock_face_detector.detect_faces.return_value = []  # No faces
        
        mock_mask_detector = Mock()
        mock_helmet_detector = Mock()
        
        mock_security_manager = Mock()
        mock_security_manager.get_status.return_value = {
            'access_granted': False,
            'violation_type': 'no_face_detected',
            'face_count': 0,
            'has_mask': False,
            'has_helmet': False,
            'mask_confidence': 0.0,
            'helmet_confidence': 0.0
        }
        
        # Generate frames
        frame_count = 0
        for frame_data in self.handler.generate_frames(
            mock_camera,
            mock_face_detector,
            mock_mask_detector,
            mock_helmet_detector,
            mock_security_manager
        ):
            self.assertIsInstance(frame_data, bytes)
            frame_count += 1
        
        self.assertEqual(frame_count, 1)
        
        # Verify security manager was updated with no faces
        mock_security_manager.update_status.assert_called_once()
        call_args = mock_security_manager.update_status.call_args[0]
        self.assertEqual(call_args[0], 0)  # face_count should be 0
    
    def test_generate_frames_with_mask_detection(self):
        """Test frame generation with mask detected."""
        # Create mock camera
        mock_camera = Mock()
        mock_camera.read.side_effect = [
            (True, self.test_frame.copy()),
            (False, None)
        ]
        
        # Create mock detectors
        mock_face_detector = Mock()
        mock_face_detector.detect_faces.return_value = [(100, 100, 200, 200)]
        mock_face_detector.extract_face_roi.return_value = np.ones((200, 200, 3), dtype=np.uint8) * 128
        mock_face_detector.preprocess_face.return_value = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        mock_mask_detector = Mock()
        mock_mask_detector.detect_mask.return_value = (True, 0.75)  # Mask detected
        
        mock_helmet_detector = Mock()
        mock_helmet_detector.detect_helmet.return_value = (False, 0.1)
        
        mock_security_manager = Mock()
        mock_security_manager.get_status.return_value = self.security_status_denied
        
        # Generate frames
        frame_count = 0
        for frame_data in self.handler.generate_frames(
            mock_camera,
            mock_face_detector,
            mock_mask_detector,
            mock_helmet_detector,
            mock_security_manager
        ):
            self.assertIsInstance(frame_data, bytes)
            frame_count += 1
        
        self.assertEqual(frame_count, 1)
        
        # Verify mask detector was called
        mock_mask_detector.detect_mask.assert_called_once()
        
        # Verify security manager was updated with mask detection
        mock_security_manager.update_status.assert_called_once()
        call_args = mock_security_manager.update_status.call_args[0]
        self.assertEqual(call_args[1], True)  # has_mask should be True


if __name__ == '__main__':
    unittest.main()
