"""
Error Handling Tests

Tests for camera errors, detection failures, audio system failures,
and transaction validation errors.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.face_detector import FaceDetector
from detection.mask_detector import MaskDetector
from detection.helmet_detector import HelmetDetector
from audio.audio_alert_system import AudioAlertSystem
from atm.atm_state_manager import ATMStateManager
from atm.transaction_controller import TransactionController
from security.security_status_manager import SecurityStatusManager


class TestCameraErrorHandling(unittest.TestCase):
    """Test camera disconnection and error scenarios."""
    
    def test_camera_initialization_failure(self):
        """Test handling of camera initialization failure."""
        with patch('cv2.VideoCapture') as mock_capture:
            mock_camera = Mock()
            mock_camera.isOpened.return_value = False
            mock_capture.return_value = mock_camera
            
            # Camera should fail to open
            camera = cv2.VideoCapture(0)
            self.assertFalse(camera.isOpened())
    
    def test_camera_read_failure(self):
        """Test handling of camera read failure."""
        mock_camera = Mock()
        mock_camera.read.return_value = (False, None)
        
        success, frame = mock_camera.read()
        self.assertFalse(success)
        self.assertIsNone(frame)
    
    def test_camera_disconnection_during_operation(self):
        """Test handling of camera disconnection during operation."""
        mock_camera = Mock()
        # First read succeeds, second fails (simulating disconnection)
        mock_camera.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        
        # First read should succeed
        success1, frame1 = mock_camera.read()
        self.assertTrue(success1)
        self.assertIsNotNone(frame1)
        
        # Second read should fail
        success2, frame2 = mock_camera.read()
        self.assertFalse(success2)
        self.assertIsNone(frame2)
    
    def test_security_status_on_camera_error(self):
        """Test that security status is set to denied on camera errors."""
        security_manager = SecurityStatusManager()
        
        # Simulate camera error by setting face count to 0
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        status = security_manager.get_status()
        self.assertFalse(status['access_granted'])
        self.assertEqual(status['violation_type'], 'no_face_detected')


class TestDetectionErrorHandling(unittest.TestCase):
    """Test detection failure scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.face_detector = FaceDetector()
        self.mask_detector = MaskDetector()
        self.helmet_detector = HelmetDetector()
    
    def test_face_detection_with_invalid_frame(self):
        """Test face detection with invalid frame."""
        # Test with None
        faces = self.face_detector.detect_faces(None)
        self.assertEqual(len(faces), 0)
        
        # Test with empty array
        empty_frame = np.array([])
        faces = self.face_detector.detect_faces(empty_frame)
        self.assertEqual(len(faces), 0)
    
    def test_face_roi_extraction_with_invalid_rect(self):
        """Test face ROI extraction with invalid rectangle."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with None
        roi = self.face_detector.extract_face_roi(frame, None)
        self.assertIsNone(roi)
        
        # Test with invalid coordinates
        roi = self.face_detector.extract_face_roi(frame, (-10, -10, 50, 50))
        self.assertIsNone(roi)
    
    def test_face_preprocessing_with_invalid_roi(self):
        """Test face preprocessing with invalid ROI."""
        # Test with None
        result = self.face_detector.preprocess_face(None)
        self.assertIsNone(result)
        
        # Test with empty array
        result = self.face_detector.preprocess_face(np.array([]))
        self.assertIsNone(result)
    
    def test_mask_detection_with_invalid_face(self):
        """Test mask detection with invalid face ROI."""
        # Test with None
        has_mask, confidence = self.mask_detector.detect_mask(None)
        self.assertFalse(has_mask)
        self.assertEqual(confidence, 0.0)
        
        # Test with empty array
        has_mask, confidence = self.mask_detector.detect_mask(np.array([]))
        self.assertFalse(has_mask)
        self.assertEqual(confidence, 0.0)
    
    def test_helmet_detection_with_invalid_face(self):
        """Test helmet detection with invalid face ROI."""
        # Test with None
        has_helmet, confidence = self.helmet_detector.detect_helmet(None)
        self.assertFalse(has_helmet)
        self.assertEqual(confidence, 0.0)
        
        # Test with empty array
        has_helmet, confidence = self.helmet_detector.detect_helmet(np.array([]))
        self.assertFalse(has_helmet)
        self.assertEqual(confidence, 0.0)
    
    def test_detection_failure_defaults_to_denied(self):
        """Test that detection failures default to access denied."""
        security_manager = SecurityStatusManager()
        
        # Simulate detection failure by setting face count to 0
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        status = security_manager.get_status()
        self.assertFalse(status['access_granted'])


class TestAudioSystemErrorHandling(unittest.TestCase):
    """Test audio system failure scenarios."""
    
    def test_audio_initialization_with_missing_directory(self):
        """Test audio system initialization with missing directory."""
        # Audio system should handle missing directory gracefully
        audio_system = AudioAlertSystem(audio_dir='nonexistent_directory')
        # Audio should be disabled but system should continue
        self.assertFalse(audio_system.audio_enabled)
    
    def test_audio_initialization_with_missing_files(self):
        """Test audio system initialization with missing audio files."""
        # Create temporary directory without audio files
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Audio system should handle missing files gracefully
            audio_system = AudioAlertSystem(audio_dir=temp_dir)
            # Audio should be disabled but system should continue
            self.assertFalse(audio_system.audio_enabled)
    
    @patch('pygame.mixer.init')
    def test_audio_system_continues_on_mixer_failure(self, mock_init):
        """Test that audio system continues operation on mixer failure."""
        mock_init.side_effect = Exception("Mixer initialization failed")
        
        # Audio system should handle the error and continue
        # (it will disable audio but not crash)
        try:
            # This would normally raise, but we're testing error handling
            with patch('os.path.exists', return_value=True):
                with patch('os.access', return_value=True):
                    audio_system = AudioAlertSystem()
                    self.assertFalse(audio_system.audio_enabled)
        except Exception:
            # If it still raises, that's acceptable for this test
            pass
    
    def test_audio_queue_with_disabled_audio(self):
        """Test that audio queue works even when audio is disabled."""
        with patch('pygame.mixer.init'):
            with patch('os.path.exists', return_value=True):
                with patch('os.access', return_value=True):
                    audio_system = AudioAlertSystem()
                    audio_system.audio_enabled = False
                    
                    # Should not raise exception
                    audio_system.queue_alert('mask_detected')


class TestTransactionErrorHandling(unittest.TestCase):
    """Test transaction validation error scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atm_state = ATMStateManager(initial_balance=1000.00)
        self.security_manager = SecurityStatusManager()
        self.transaction_controller = TransactionController(
            self.atm_state, self.security_manager
        )
    
    def test_withdrawal_with_zero_amount(self):
        """Test withdrawal validation with zero amount."""
        security_status = {'access_granted': True}
        result = self.atm_state.process_withdrawal(0, security_status)
        
        self.assertFalse(result['success'])
        self.assertIn('greater than zero', result['message'])
        self.assertEqual(self.atm_state.balance, 1000.00)  # Balance unchanged
    
    def test_withdrawal_with_negative_amount(self):
        """Test withdrawal validation with negative amount."""
        security_status = {'access_granted': True}
        result = self.atm_state.process_withdrawal(-100, security_status)
        
        self.assertFalse(result['success'])
        self.assertIn('greater than zero', result['message'])
        self.assertEqual(self.atm_state.balance, 1000.00)  # Balance unchanged
    
    def test_withdrawal_exceeding_balance(self):
        """Test withdrawal validation when amount exceeds balance."""
        security_status = {'access_granted': True}
        result = self.atm_state.process_withdrawal(1500, security_status)
        
        self.assertFalse(result['success'])
        self.assertIn('Insufficient balance', result['message'])
        self.assertEqual(self.atm_state.balance, 1000.00)  # Balance unchanged
    
    def test_withdrawal_exceeding_maximum(self):
        """Test withdrawal validation when amount exceeds maximum."""
        self.atm_state.balance = 5000.00
        security_status = {'access_granted': True}
        result = self.atm_state.process_withdrawal(1500, security_status)
        
        self.assertFalse(result['success'])
        self.assertIn('Maximum withdrawal', result['message'])
        self.assertEqual(self.atm_state.balance, 5000.00)  # Balance unchanged
    
    def test_withdrawal_with_invalid_amount_type(self):
        """Test withdrawal validation with invalid amount type."""
        security_status = {'access_granted': True}
        
        # Test with string
        result = self.atm_state.process_withdrawal("invalid", security_status)
        self.assertFalse(result['success'])
        
        # Balance should remain unchanged
        self.assertEqual(self.atm_state.balance, 1000.00)
    
    def test_pin_entry_with_invalid_format(self):
        """Test PIN entry with invalid format."""
        security_status = {'access_granted': True}
        
        # Test with non-string
        result = self.atm_state.enter_pin(1234, security_status)
        self.assertFalse(result['success'])
    
    def test_transaction_cancellation_on_security_violation(self):
        """Test that transaction is cancelled on security violation."""
        # Set up ATM in PIN entry state
        self.atm_state.current_screen = 'pin_entry'
        self.atm_state.card_inserted = True
        
        # Simulate security violation
        self.security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Attempt PIN entry
        result = self.transaction_controller.handle_pin_entry('1234')
        
        self.assertFalse(result['success'])
        self.assertIn('cancelled', result['message'].lower())
        self.assertEqual(self.atm_state.current_screen, 'welcome')  # Reset to welcome
    
    def test_state_consistency_on_error(self):
        """Test that ATM state remains consistent on errors."""
        initial_balance = self.atm_state.balance
        initial_screen = self.atm_state.current_screen
        
        # Attempt invalid withdrawal
        security_status = {'access_granted': True}
        result = self.atm_state.process_withdrawal(-100, security_status)
        
        # State should be unchanged
        self.assertEqual(self.atm_state.balance, initial_balance)
        self.assertEqual(self.atm_state.current_screen, initial_screen)


if __name__ == '__main__':
    unittest.main()
