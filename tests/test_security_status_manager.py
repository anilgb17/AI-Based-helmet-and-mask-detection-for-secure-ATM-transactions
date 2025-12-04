"""
Unit tests for Security Status Manager Module

Tests the SecurityStatusManager class functionality including:
- All violation scenarios (no face, multiple faces, mask, helmet, both)
- Access granted scenario
- Warning counter reset
"""

import unittest
import sys
import os

# Add parent directory to path to import security module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from security.security_status_manager import SecurityStatusManager


class TestSecurityStatusManager(unittest.TestCase):
    """Test cases for SecurityStatusManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.manager = SecurityStatusManager()
    
    def test_initialization(self):
        """Test SecurityStatusManager initialization with default values."""
        self.assertFalse(self.manager.access_granted)
        self.assertEqual(self.manager.violation_type, "no_face_detected")
        self.assertEqual(self.manager.face_count, 0)
        self.assertFalse(self.manager.has_mask)
        self.assertFalse(self.manager.has_helmet)
        self.assertEqual(self.manager.mask_confidence, 0.0)
        self.assertEqual(self.manager.helmet_confidence, 0.0)
        self.assertEqual(self.manager.warning_count, 0)
        self.assertIsNotNone(self.manager.timestamp)
    
    def test_no_face_detected(self):
        """Test security status when no face is detected."""
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        self.assertFalse(self.manager.access_granted)
        self.assertEqual(self.manager.violation_type, "no_face_detected")
        self.assertEqual(self.manager.face_count, 0)
        self.assertEqual(self.manager.warning_count, 1)
    
    def test_multiple_people_detected(self):
        """Test security status when multiple faces are detected."""
        self.manager.update_status(
            face_count=3,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        self.assertFalse(self.manager.access_granted)
        self.assertEqual(self.manager.violation_type, "multiple_people")
        self.assertEqual(self.manager.face_count, 3)
        self.assertEqual(self.manager.warning_count, 1)
    
    def test_mask_detected(self):
        """Test security status when mask is detected."""
        self.manager.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.75, 'helmet': 0.2}
        )
        
        self.assertFalse(self.manager.access_granted)
        self.assertEqual(self.manager.violation_type, "mask_detected")
        self.assertEqual(self.manager.face_count, 1)
        self.assertTrue(self.manager.has_mask)
        self.assertFalse(self.manager.has_helmet)
        self.assertEqual(self.manager.mask_confidence, 0.75)
        self.assertEqual(self.manager.warning_count, 1)
    
    def test_helmet_detected(self):
        """Test security status when helmet is detected."""
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=True,
            confidences={'mask': 0.2, 'helmet': 0.65}
        )
        
        self.assertFalse(self.manager.access_granted)
        self.assertEqual(self.manager.violation_type, "helmet_detected")
        self.assertEqual(self.manager.face_count, 1)
        self.assertFalse(self.manager.has_mask)
        self.assertTrue(self.manager.has_helmet)
        self.assertEqual(self.manager.helmet_confidence, 0.65)
        self.assertEqual(self.manager.warning_count, 1)
    
    def test_mask_and_helmet_detected(self):
        """Test security status when both mask and helmet are detected."""
        self.manager.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=True,
            confidences={'mask': 0.8, 'helmet': 0.7}
        )
        
        self.assertFalse(self.manager.access_granted)
        self.assertEqual(self.manager.violation_type, "mask_and_helmet_detected")
        self.assertEqual(self.manager.face_count, 1)
        self.assertTrue(self.manager.has_mask)
        self.assertTrue(self.manager.has_helmet)
        self.assertEqual(self.manager.mask_confidence, 0.8)
        self.assertEqual(self.manager.helmet_confidence, 0.7)
        self.assertEqual(self.manager.warning_count, 1)
    
    def test_access_granted(self):
        """Test security status when access should be granted."""
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.15}
        )
        
        self.assertTrue(self.manager.access_granted)
        self.assertIsNone(self.manager.violation_type)
        self.assertEqual(self.manager.face_count, 1)
        self.assertFalse(self.manager.has_mask)
        self.assertFalse(self.manager.has_helmet)
        self.assertEqual(self.manager.warning_count, 0)
    
    def test_warning_counter_increment(self):
        """Test that warning counter increments on consecutive violations."""
        # First violation
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        self.assertEqual(self.manager.warning_count, 1)
        
        # Second violation
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        self.assertEqual(self.manager.warning_count, 2)
        
        # Third violation
        self.manager.update_status(
            face_count=2,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        self.assertEqual(self.manager.warning_count, 3)
    
    def test_warning_counter_reset_on_access_granted(self):
        """Test that warning counter resets when access is granted."""
        # Create some violations
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        self.assertEqual(self.manager.warning_count, 2)
        
        # Grant access
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.1}
        )
        
        # Warning counter should be reset
        self.assertEqual(self.manager.warning_count, 0)
    
    def test_reset_warnings_method(self):
        """Test the reset_warnings method."""
        # Create violations
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        self.assertEqual(self.manager.warning_count, 1)
        
        # Manually reset warnings
        self.manager.reset_warnings()
        self.assertEqual(self.manager.warning_count, 0)
    
    def test_get_status_method(self):
        """Test the get_status method returns complete status dictionary."""
        self.manager.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.85, 'helmet': 0.25}
        )
        
        status = self.manager.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('access_granted', status)
        self.assertIn('violation_type', status)
        self.assertIn('face_count', status)
        self.assertIn('has_mask', status)
        self.assertIn('has_helmet', status)
        self.assertIn('mask_confidence', status)
        self.assertIn('helmet_confidence', status)
        self.assertIn('warning_count', status)
        self.assertIn('timestamp', status)
        
        self.assertFalse(status['access_granted'])
        self.assertEqual(status['violation_type'], 'mask_detected')
        self.assertEqual(status['face_count'], 1)
        self.assertTrue(status['has_mask'])
        self.assertFalse(status['has_helmet'])
        self.assertEqual(status['mask_confidence'], 0.85)
        self.assertEqual(status['helmet_confidence'], 0.25)
    
    def test_get_violation_message_no_face(self):
        """Test violation message for no face detected."""
        self.manager.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        message = self.manager.get_violation_message()
        self.assertIsNotNone(message)
        self.assertIn('No face detected', message)
    
    def test_get_violation_message_multiple_people(self):
        """Test violation message for multiple people."""
        self.manager.update_status(
            face_count=2,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        message = self.manager.get_violation_message()
        self.assertIsNotNone(message)
        self.assertIn('Multiple people', message)
    
    def test_get_violation_message_mask(self):
        """Test violation message for mask detected."""
        self.manager.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.7, 'helmet': 0.2}
        )
        
        message = self.manager.get_violation_message()
        self.assertIsNotNone(message)
        self.assertIn('mask', message.lower())
    
    def test_get_violation_message_helmet(self):
        """Test violation message for helmet detected."""
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=True,
            confidences={'mask': 0.2, 'helmet': 0.6}
        )
        
        message = self.manager.get_violation_message()
        self.assertIsNotNone(message)
        self.assertIn('helmet', message.lower())
    
    def test_get_violation_message_both(self):
        """Test violation message for both mask and helmet."""
        self.manager.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=True,
            confidences={'mask': 0.8, 'helmet': 0.7}
        )
        
        message = self.manager.get_violation_message()
        self.assertIsNotNone(message)
        self.assertIn('mask', message.lower())
        self.assertIn('helmet', message.lower())
    
    def test_get_violation_message_access_granted(self):
        """Test violation message when access is granted."""
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.1}
        )
        
        message = self.manager.get_violation_message()
        self.assertIsNone(message)
    
    def test_confidence_scores_stored(self):
        """Test that confidence scores are properly stored."""
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.35, 'helmet': 0.28}
        )
        
        self.assertEqual(self.manager.mask_confidence, 0.35)
        self.assertEqual(self.manager.helmet_confidence, 0.28)
    
    def test_timestamp_updated(self):
        """Test that timestamp is updated on status update."""
        initial_timestamp = self.manager.timestamp
        
        import time
        time.sleep(0.01)  # Small delay
        
        self.manager.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        self.assertGreater(self.manager.timestamp, initial_timestamp)


if __name__ == '__main__':
    unittest.main()
