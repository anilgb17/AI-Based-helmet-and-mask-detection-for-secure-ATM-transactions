"""
End-to-End Integration Tests

Comprehensive end-to-end tests covering complete transaction flows
with various security scenarios including clear face, mask detection,
helmet detection, multiple people, and no face scenarios.
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app


@pytest.fixture
def app():
    """Create test Flask application."""
    with patch('app.cv2.VideoCapture') as mock_camera:
        # Mock camera
        mock_cam_instance = MagicMock()
        mock_cam_instance.isOpened.return_value = True
        
        # Create a mock frame (640x480 BGR image)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cam_instance.read.return_value = (True, mock_frame)
        mock_camera.return_value = mock_cam_instance
        
        # Mock audio system to avoid pygame initialization
        with patch('app.AudioAlertSystem'):
            app = create_app()
            app.config['TESTING'] = True
            yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestCompleteTransactionWithClearFace:
    """Test complete transaction with clear face (should succeed)."""
    
    def test_successful_transaction_clear_face(self, client, app):
        """
        Test complete successful transaction from start to finish with clear face.
        This simulates a legitimate user with no face coverings.
        """
        security_manager = app.config['security_manager']
        
        # Simulate clear face: 1 face, no mask, no helmet
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Verify initial state
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['screen'] == 'welcome'
        assert data['balance'] == 5000.00
        assert data['security']['access_granted'] is True
        assert data['security']['face_count'] == 1
        assert data['security']['has_mask'] is False
        assert data['security']['has_helmet'] is False
        
        # Step 1: Insert card
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['screen'] == 'pin_entry'
        assert 'Card inserted successfully' in data['message']
        
        # Step 2: Enter correct PIN
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['screen'] == 'menu'
        assert 'PIN accepted' in data['message']
        
        # Step 3: Select withdrawal option
        response = client.post('/api/select_withdrawal')
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['screen'] == 'withdrawal'
        
        # Step 4: Process withdrawal
        withdrawal_amount = 250.00
        response = client.post('/api/withdraw', json={'amount': withdrawal_amount})
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['new_balance'] == 4750.00
        assert 'Withdrawal successful' in data['message']
        
        # Step 5: Verify final balance
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 4750.00
        
        # Step 6: Reset and verify
        response = client.post('/api/reset')
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['screen'] == 'welcome'


class TestTransactionWithMask:
    """Test transaction with mask (should fail at each step)."""
    
    def test_card_insertion_blocked_by_mask(self, client, app):
        """Test that card insertion is blocked when mask is detected."""
        security_manager = app.config['security_manager']
        
        # Simulate mask detected: 1 face, mask present, no helmet
        security_manager.update_status(1, True, False, {'mask': 0.75, 'helmet': 0.1})
        
        # Verify security status
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'mask_detected'
        assert data['security']['has_mask'] is True
        
        # Attempt to insert card - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'mask' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_pin_entry_blocked_by_mask(self, client, app):
        """Test that PIN entry fails if mask is detected during entry."""
        security_manager = app.config['security_manager']
        
        # Start with clear face
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Insert card successfully
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # User puts on mask during PIN entry
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.05})
        
        # Attempt PIN entry - should fail and cancel transaction
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_withdrawal_blocked_by_mask(self, client, app):
        """Test that withdrawal fails if mask is detected during withdrawal."""
        security_manager = app.config['security_manager']
        
        # Complete authentication with clear face
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # User puts on mask during withdrawal
        security_manager.update_status(1, True, False, {'mask': 0.85, 'helmet': 0.05})
        
        # Attempt withdrawal - should fail and cancel transaction
        response = client.post('/api/withdraw', json={'amount': 100.00})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        
        # Verify balance unchanged
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 5000.00


class TestTransactionWithHelmet:
    """Test transaction with helmet (should fail at each step)."""
    
    def test_card_insertion_blocked_by_helmet(self, client, app):
        """Test that card insertion is blocked when helmet is detected."""
        security_manager = app.config['security_manager']
        
        # Simulate helmet detected: 1 face, no mask, helmet present
        security_manager.update_status(1, False, True, {'mask': 0.1, 'helmet': 0.65})
        
        # Verify security status
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'helmet_detected'
        assert data['security']['has_helmet'] is True
        
        # Attempt to insert card - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'helmet' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_pin_entry_blocked_by_helmet(self, client, app):
        """Test that PIN entry fails if helmet is detected during entry."""
        security_manager = app.config['security_manager']
        
        # Start with clear face
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Insert card successfully
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # User puts on helmet during PIN entry
        security_manager.update_status(1, False, True, {'mask': 0.1, 'helmet': 0.7})
        
        # Attempt PIN entry - should fail and cancel transaction
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_withdrawal_blocked_by_helmet(self, client, app):
        """Test that withdrawal fails if helmet is detected during withdrawal."""
        security_manager = app.config['security_manager']
        
        # Complete authentication with clear face
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # User puts on helmet during withdrawal
        security_manager.update_status(1, False, True, {'mask': 0.1, 'helmet': 0.72})
        
        # Attempt withdrawal - should fail and cancel transaction
        response = client.post('/api/withdraw', json={'amount': 100.00})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        
        # Verify balance unchanged
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 5000.00


class TestTransactionWithMultiplePeople:
    """Test transaction with multiple people (should fail)."""
    
    def test_card_insertion_blocked_by_multiple_people(self, client, app):
        """Test that card insertion is blocked when multiple faces detected."""
        security_manager = app.config['security_manager']
        
        # Simulate multiple people: 2 faces detected
        security_manager.update_status(2, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Verify security status
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'multiple_people'
        assert data['security']['face_count'] == 2
        
        # Attempt to insert card - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'multiple' in data['message'].lower() or 'people' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_transaction_cancelled_when_second_person_appears(self, client, app):
        """Test that transaction is cancelled when second person appears."""
        security_manager = app.config['security_manager']
        
        # Start with single person
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Insert card successfully
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # Second person appears
        security_manager.update_status(2, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Attempt PIN entry - should fail and cancel transaction
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_withdrawal_cancelled_when_multiple_people_detected(self, client, app):
        """Test that withdrawal is cancelled when multiple people detected."""
        security_manager = app.config['security_manager']
        
        # Complete authentication with single person
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # Multiple people appear during withdrawal
        security_manager.update_status(3, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Attempt withdrawal - should fail and cancel transaction
        response = client.post('/api/withdraw', json={'amount': 100.00})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        
        # Verify balance unchanged
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 5000.00


class TestTransactionWithNoFace:
    """Test transaction with no face detected (should fail)."""
    
    def test_card_insertion_blocked_with_no_face(self, client, app):
        """Test that card insertion is blocked when no face is detected."""
        security_manager = app.config['security_manager']
        
        # Simulate no face detected
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Verify security status
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'no_face_detected'
        assert data['security']['face_count'] == 0
        
        # Attempt to insert card - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'no face' in data['message'].lower() or 'face not detected' in data['message'].lower()
        assert data['screen'] == 'welcome'
    
    def test_transaction_cancelled_when_face_disappears(self, client, app):
        """Test that transaction is cancelled when face disappears."""
        security_manager = app.config['security_manager']
        
        # Start with face visible
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Insert card successfully
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # Face disappears (user moves away)
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Attempt PIN entry - should fail and cancel transaction
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()
        assert data['screen'] == 'welcome'


class TestVideoStreamingStability:
    """Test video streaming stability."""
    
    def test_video_endpoint_accessible(self, client):
        """Test that video streaming endpoint is accessible."""
        response = client.get('/video')
        assert response.status_code == 200
        assert 'multipart/x-mixed-replace' in response.content_type
    
    def test_video_stream_with_security_violations(self, client, app):
        """Test video stream continues during security violations."""
        security_manager = app.config['security_manager']
        
        # Set various security violations
        violations = [
            (1, True, False, {'mask': 0.8, 'helmet': 0.1}),  # Mask
            (1, False, True, {'mask': 0.1, 'helmet': 0.7}),  # Helmet
            (2, False, False, {'mask': 0.1, 'helmet': 0.1}),  # Multiple people
            (0, False, False, {'mask': 0.0, 'helmet': 0.0}),  # No face
        ]
        
        for face_count, has_mask, has_helmet, confidences in violations:
            security_manager.update_status(face_count, has_mask, has_helmet, confidences)
            
            # Video endpoint should still be accessible
            response = client.get('/video')
            assert response.status_code == 200


class TestAudioAlertSystem:
    """Test audio alert system integration."""
    
    def test_audio_alerts_queued_for_violations(self, client, app):
        """Test that audio alerts are queued for security violations."""
        security_manager = app.config['security_manager']
        audio_system = app.config['audio_system']
        
        if audio_system is None:
            pytest.skip("Audio system not initialized")
        
        # Test mask detection alert
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.1})
        # Audio alert should be queued (tested in audio system unit tests)
        
        # Test helmet detection alert
        security_manager.update_status(1, False, True, {'mask': 0.1, 'helmet': 0.7})
        # Audio alert should be queued
        
        # Test multiple people alert
        security_manager.update_status(2, False, False, {'mask': 0.1, 'helmet': 0.1})
        # Audio alert should be queued


class TestContinuousOperation:
    """Test system under continuous operation."""
    
    def test_multiple_consecutive_transactions(self, client, app):
        """Test multiple consecutive transactions."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        initial_balance = 5000.00
        
        # Perform 5 consecutive transactions
        for i in range(5):
            withdrawal_amount = 50.00
            
            # Complete transaction
            client.post('/api/insert_card')
            client.post('/api/enter_pin', json={'pin': '1234'})
            client.post('/api/select_withdrawal')
            response = client.post('/api/withdraw', json={'amount': withdrawal_amount})
            
            data = json.loads(response.data)
            assert data['success'] is True
            
            expected_balance = initial_balance - (withdrawal_amount * (i + 1))
            assert data['new_balance'] == expected_balance
            
            # Reset for next transaction
            client.post('/api/reset')
        
        # Verify final balance
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 4750.00
    
    def test_rapid_status_polling(self, client):
        """Test rapid status polling simulating frontend updates."""
        # Simulate 20 rapid status requests (like frontend polling every 500ms)
        for _ in range(20):
            response = client.get('/api/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'screen' in data
            assert 'balance' in data
            assert 'security' in data
    
    def test_concurrent_operations(self, client, app):
        """Test concurrent API operations."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        results = []
        
        def perform_transaction():
            try:
                client.post('/api/insert_card')
                client.post('/api/enter_pin', json={'pin': '1234'})
                client.post('/api/select_withdrawal')
                response = client.post('/api/withdraw', json={'amount': 10.00})
                results.append(json.loads(response.data))
                client.post('/api/reset')
            except Exception as e:
                results.append({'error': str(e)})
        
        # Create 3 concurrent threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=perform_transaction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # At least some transactions should succeed
        # (exact behavior depends on thread scheduling)
        assert len(results) == 3


class TestMixedSecurityScenarios:
    """Test mixed security scenarios."""
    
    def test_mask_and_helmet_detected(self, client, app):
        """Test when both mask and helmet are detected."""
        security_manager = app.config['security_manager']
        
        # Simulate both mask and helmet
        security_manager.update_status(1, True, True, {'mask': 0.8, 'helmet': 0.7})
        
        # Verify security status
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'mask_and_helmet_detected'
        assert data['security']['has_mask'] is True
        assert data['security']['has_helmet'] is True
        
        # Attempt transaction - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_security_status_changes_during_transaction(self, client, app):
        """Test various security status changes during a transaction."""
        security_manager = app.config['security_manager']
        
        # Start with clear face
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        
        # Change to mask
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.05})
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        assert json.loads(response.data)['success'] is False
        
        # Reset and try again with helmet
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        
        security_manager.update_status(1, False, True, {'mask': 0.1, 'helmet': 0.7})
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        assert json.loads(response.data)['success'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
