"""Unit tests for AudioAlertSystem."""

import unittest
import os
import time
import tempfile
import shutil
from audio.audio_alert_system import AudioAlertSystem


class TestAudioAlertSystem(unittest.TestCase):
    """Test cases for AudioAlertSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use the actual audio directory for tests
        self.audio_dir = 'audio'
        
    def test_audio_file_validation_success(self):
        """Test that audio file validation succeeds when all files exist."""
        # Should not raise an exception
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        system.shutdown()
    
    def test_audio_file_validation_missing_directory(self):
        """Test that audio file validation fails when directory doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            AudioAlertSystem(audio_dir='nonexistent_directory')
    
    def test_audio_file_validation_missing_files(self):
        """Test that audio file validation fails when required files are missing."""
        # Create temporary directory with missing files
        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                AudioAlertSystem(audio_dir=temp_dir)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_audio_file_mapping(self):
        """Test audio file path mapping for different violation types."""
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        
        # Test valid violation types
        self.assertEqual(
            system._get_audio_file('mask_detected'),
            os.path.join(self.audio_dir, 'mask_detected.mp3')
        )
        self.assertEqual(
            system._get_audio_file('helmet_detected'),
            os.path.join(self.audio_dir, 'helmet_detected.mp3')
        )
        self.assertEqual(
            system._get_audio_file('mask_and_helmet_detected'),
            os.path.join(self.audio_dir, 'mask_and_helmet.mp3')
        )
        self.assertEqual(
            system._get_audio_file('multiple_people'),
            os.path.join(self.audio_dir, 'multiple_people.mp3')
        )
        
        # Test invalid violation type
        self.assertIsNone(system._get_audio_file('invalid_type'))
        
        system.shutdown()
    
    def test_queue_alert(self):
        """Test message queueing functionality."""
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        
        # Queue should be empty initially
        self.assertTrue(system.message_queue.empty())
        
        # Queue an alert
        system.queue_alert('mask_detected')
        
        # Queue should have one item
        self.assertFalse(system.message_queue.empty())
        self.assertEqual(system.message_queue.qsize(), 1)
        
        # Queue another alert
        system.queue_alert('helmet_detected')
        self.assertEqual(system.message_queue.qsize(), 2)
        
        system.shutdown()
    
    def test_queue_alert_invalid_type(self):
        """Test that invalid violation types are not queued."""
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        
        # Queue an invalid alert type
        system.queue_alert('invalid_type')
        
        # Queue should remain empty
        self.assertTrue(system.message_queue.empty())
        
        system.shutdown()
    
    def test_rate_limiting(self):
        """Test that rate limiting prevents alerts from playing too frequently."""
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        
        # Set a shorter rate limit for testing
        system.rate_limit_seconds = 2
        
        # Queue multiple alerts
        system.queue_alert('mask_detected')
        system.queue_alert('helmet_detected')
        
        # Wait for first alert to process
        time.sleep(0.5)
        
        # Check that last_alert_time was updated
        first_alert_time = system.last_alert_time
        self.assertGreater(first_alert_time, 0)
        
        # Wait less than rate limit
        time.sleep(1)
        
        # Second alert should be rate limited (not enough time passed)
        # The last_alert_time should not have changed significantly
        time.sleep(0.5)
        
        # Now wait for rate limit to expire and process second alert
        time.sleep(1.5)
        
        # Verify rate limiting is working by checking time difference
        self.assertGreaterEqual(
            system.last_alert_time - first_alert_time,
            system.rate_limit_seconds - 0.5  # Allow small tolerance
        )
        
        system.shutdown()
    
    def test_worker_thread_starts(self):
        """Test that worker thread starts successfully."""
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        
        # Worker thread should be running
        self.assertTrue(system.running)
        self.assertIsNotNone(system.worker_thread)
        self.assertTrue(system.worker_thread.is_alive())
        
        system.shutdown()
    
    def test_shutdown(self):
        """Test graceful shutdown of audio system."""
        system = AudioAlertSystem(audio_dir=self.audio_dir)
        
        # Verify system is running
        self.assertTrue(system.running)
        self.assertTrue(system.worker_thread.is_alive())
        
        # Shutdown
        system.shutdown()
        
        # Verify system stopped
        self.assertFalse(system.running)
        
        # Wait a bit for thread to finish
        time.sleep(0.5)
        
        # Thread should no longer be alive
        self.assertFalse(system.worker_thread.is_alive())


if __name__ == '__main__':
    unittest.main()
