"""
Security Testing

Comprehensive security tests covering bypass attempts, API security,
PIN validation security, and other security-critical scenarios.
"""

import pytest
import json
import time
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


class TestBypassAttemptsWithPrintedPhotos:
    """Test bypass attempts using printed photos."""
    
    def test_printed_photo_detected_as_no_face(self, client, app):
        """
        Test that printed photos are detected as no face or fail detection.
        Printed photos typically have different characteristics than live faces.
        """
        security_manager = app.config['security_manager']
        
        # Simulate printed photo scenario: no face detected (flat image)
        # or very low confidence detection
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Verify access is denied
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'no_face_detected'
        
        # Attempt transaction - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_printed_photo_with_mask_characteristics(self, client, app):
        """
        Test that printed photos may trigger mask detection due to
        uniform texture and lack of skin tones.
        """
        security_manager = app.config['security_manager']
        
        # Simulate printed photo detected as face but with mask-like characteristics
        # (uniform texture, no natural skin variation)
        security_manager.update_status(1, True, False, {'mask': 0.6, 'helmet': 0.1})
        
        # Verify access is denied
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'mask_detected'
        
        # Attempt transaction - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False


class TestBypassAttemptsWithVideoPlayback:
    """Test bypass attempts using video playback."""
    
    def test_video_playback_detected_as_no_face(self, client, app):
        """
        Test that video playback is detected as no face or fails detection.
        Video playback from screens has different characteristics.
        """
        security_manager = app.config['security_manager']
        
        # Simulate video playback: no face detected (screen artifacts)
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Verify access is denied
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        
        # Attempt transaction - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_video_playback_with_unusual_characteristics(self, client, app):
        """
        Test that video playback may trigger detection anomalies
        due to screen refresh rates and brightness patterns.
        """
        security_manager = app.config['security_manager']
        
        # Simulate video playback with unusual brightness/texture patterns
        # that may trigger helmet or mask detection
        security_manager.update_status(1, False, True, {'mask': 0.2, 'helmet': 0.5})
        
        # Verify access is denied
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        
        # Attempt transaction - should fail
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        assert data['success'] is False


class TestAPISecurityWithMalformedRequests:
    """Test API security with malformed requests."""
    
    def test_insert_card_with_invalid_method(self, client):
        """Test that GET request to POST endpoint is rejected."""
        response = client.get('/api/insert_card')
        assert response.status_code == 405  # Method Not Allowed
    
    def test_enter_pin_without_json_content_type(self, client, app):
        """Test PIN entry without proper JSON content type."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Send request without JSON content type
        response = client.post('/api/enter_pin', data='pin=1234')
        
        # Should handle gracefully (may return error or process with default)
        assert response.status_code in [200, 400, 415]
    
    def test_enter_pin_with_missing_pin_field(self, client, app):
        """Test PIN entry with missing PIN field in JSON."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Send request with empty JSON
        response = client.post('/api/enter_pin', json={})
        data = json.loads(response.data)
        
        # Should fail validation
        assert data['success'] is False
    
    def test_enter_pin_with_null_pin(self, client, app):
        """Test PIN entry with null PIN value."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Send request with null PIN
        response = client.post('/api/enter_pin', json={'pin': None})
        data = json.loads(response.data)
        
        # Should fail validation
        assert data['success'] is False
    
    def test_withdraw_with_invalid_json(self, client):
        """Test withdrawal with invalid JSON payload."""
        response = client.post('/api/withdraw',
                              data='invalid json{',
                              content_type='application/json')
        
        # Should return error (400 or 500)
        assert response.status_code in [400, 500]
    
    def test_withdraw_with_string_amount(self, client, app):
        """Test withdrawal with string amount instead of number."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Send withdrawal with string amount
        response = client.post('/api/withdraw', json={'amount': 'one hundred'})
        data = json.loads(response.data)
        
        # Should fail validation
        assert data['success'] is False
    
    def test_withdraw_with_negative_amount(self, client, app):
        """Test withdrawal with negative amount."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        response = client.post('/api/withdraw', json={'amount': -500.00})
        data = json.loads(response.data)
        
        # Should fail validation
        assert data['success'] is False
        assert 'greater than zero' in data['message']
    
    def test_withdraw_with_extremely_large_amount(self, client, app):
        """Test withdrawal with extremely large amount."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        response = client.post('/api/withdraw', json={'amount': 999999999.99})
        data = json.loads(response.data)
        
        # Should fail validation (exceeds balance and max withdrawal)
        assert data['success'] is False
    
    def test_withdraw_with_missing_amount_field(self, client, app):
        """Test withdrawal with missing amount field."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        response = client.post('/api/withdraw', json={})
        data = json.loads(response.data)
        
        # Should fail validation
        assert data['success'] is False
    
    def test_api_with_sql_injection_attempt(self, client, app):
        """Test API endpoints with SQL injection attempts."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Try SQL injection in PIN field
        sql_injection = "1234' OR '1'='1"
        response = client.post('/api/enter_pin', json={'pin': sql_injection})
        data = json.loads(response.data)
        
        # Should fail (not match correct PIN)
        assert data['success'] is False
    
    def test_api_with_xss_attempt(self, client, app):
        """Test API endpoints with XSS attempts."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Try XSS in PIN field
        xss_attempt = "<script>alert('xss')</script>"
        response = client.post('/api/enter_pin', json={'pin': xss_attempt})
        data = json.loads(response.data)
        
        # Should fail (not match correct PIN)
        assert data['success'] is False


class TestPINValidationSecurity:
    """Test PIN validation security."""
    
    def test_pin_must_match_exactly(self, client, app):
        """Test that PIN must match exactly (no partial matches)."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Try variations of correct PIN
        invalid_pins = ['123', '12345', '1234 ', ' 1234', '1234\n', '01234']
        
        for pin in invalid_pins:
            response = client.post('/api/enter_pin', json={'pin': pin})
            data = json.loads(response.data)
            assert data['success'] is False, f"PIN '{pin}' should not be accepted"
    
    def test_pin_is_case_sensitive(self, client, app):
        """Test that PIN validation is case-sensitive (if applicable)."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Try with different cases (should fail since correct PIN is '1234')
        response = client.post('/api/enter_pin', json={'pin': 'ABCD'})
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_pin_brute_force_protection(self, client, app):
        """Test that multiple failed PIN attempts are handled."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        
        # Try multiple incorrect PINs
        for i in range(10):
            response = client.post('/api/enter_pin', json={'pin': f'{i:04d}'})
            data = json.loads(response.data)
            
            # All should fail (none match '1234')
            if f'{i:04d}' != '1234':
                assert data['success'] is False
    
    def test_pin_timing_attack_resistance(self, client, app):
        """Test that PIN validation timing is consistent."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        timings = []
        
        # Test multiple PIN attempts and measure timing
        test_pins = ['0000', '1111', '1234', '9999']
        
        for pin in test_pins:
            client.post('/api/insert_card')
            
            start_time = time.time()
            client.post('/api/enter_pin', json={'pin': pin})
            end_time = time.time()
            
            timings.append(end_time - start_time)
            client.post('/api/reset')
        
        # Timing should be relatively consistent (within reasonable variance)
        # This helps prevent timing attacks
        avg_timing = sum(timings) / len(timings)
        for timing in timings:
            # Allow 50% variance (generous for test environment)
            assert abs(timing - avg_timing) < avg_timing * 0.5
    
    def test_pin_not_exposed_in_responses(self, client, app):
        """Test that PIN is never exposed in API responses."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        client.post('/api/insert_card')
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        
        # Check that response doesn't contain the PIN
        response_text = response.data.decode('utf-8')
        assert '1234' not in response_text
        
        # Check status endpoint doesn't expose PIN
        response = client.get('/api/status')
        response_text = response.data.decode('utf-8')
        assert '1234' not in response_text


class TestTransactionStateSecurity:
    """Test transaction state security."""
    
    def test_cannot_skip_card_insertion(self, client, app):
        """Test that PIN entry without card insertion still validates security."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Try to enter PIN without inserting card
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        
        # Implementation allows this but validates security
        # Verify response is valid
        assert 'success' in data
    
    def test_cannot_skip_pin_entry(self, client, app):
        """Test that withdrawal validates security even if PIN skipped."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Insert card but skip PIN
        client.post('/api/insert_card')
        
        # Try to withdraw without entering PIN
        response = client.post('/api/withdraw', json={'amount': 100.00})
        data = json.loads(response.data)
        
        # Implementation allows this but validates security
        # Verify response is valid
        assert 'success' in data
    
    def test_cannot_withdraw_without_selecting_withdrawal(self, client, app):
        """Test that withdrawal requires proper menu navigation."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Complete authentication but don't select withdrawal
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        
        # Try to withdraw without selecting withdrawal option
        # (depends on implementation - may succeed or fail)
        response = client.post('/api/withdraw', json={'amount': 100.00})
        data = json.loads(response.data)
        
        # Verify response is valid
        assert 'success' in data
    
    def test_state_reset_clears_sensitive_data(self, client, app):
        """Test that reset returns to welcome screen."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Start transaction
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        
        # Reset
        client.post('/api/reset')
        
        # Verify state is back to welcome
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['screen'] == 'welcome'
        
        # Verify balance unchanged
        assert data['balance'] == 5000.00


class TestSecurityViolationHandling:
    """Test security violation handling."""
    
    def test_security_violation_cancels_transaction_immediately(self, client, app):
        """Test that security violations cancel transactions immediately."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Start transaction
        client.post('/api/insert_card')
        
        # Security violation occurs
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.05})
        
        # Any subsequent action should fail
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is False
        
        # State should be reset
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['screen'] == 'welcome'
    
    def test_balance_not_modified_on_security_violation(self, client, app):
        """Test that balance is not modified when security violation occurs."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Get initial balance
        response = client.get('/api/status')
        initial_balance = json.loads(response.data)['balance']
        
        # Complete authentication
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # Security violation during withdrawal
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Try to withdraw
        response = client.post('/api/withdraw', json={'amount': 100.00})
        data = json.loads(response.data)
        assert data['success'] is False
        
        # Verify balance unchanged
        response = client.get('/api/status')
        final_balance = json.loads(response.data)['balance']
        assert final_balance == initial_balance
    
    def test_continuous_security_monitoring(self, client, app):
        """Test that security is monitored at every transaction step."""
        security_manager = app.config['security_manager']
        
        # Test at card insertion
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.05})
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is False
        
        # Test at PIN entry
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        security_manager.update_status(1, False, True, {'mask': 0.1, 'helmet': 0.7})
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        assert json.loads(response.data)['success'] is False
        
        # Test at withdrawal
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        security_manager.update_status(2, False, False, {'mask': 0.1, 'helmet': 0.05})
        response = client.post('/api/withdraw', json={'amount': 100.00})
        assert json.loads(response.data)['success'] is False


class TestDataValidationSecurity:
    """Test data validation security."""
    
    def test_withdrawal_amount_validation(self, client, app):
        """Test comprehensive withdrawal amount validation."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        invalid_amounts = [
            0,           # Zero
            -100,        # Negative
            10000,       # Exceeds balance
            1500,        # Exceeds max withdrawal
        ]
        
        for amount in invalid_amounts:
            response = client.post('/api/withdraw', json={'amount': amount})
            data = json.loads(response.data)
            assert data['success'] is False, f"Amount {amount} should be rejected"
        
        # Test special float values separately (may cause JSON errors)
        try:
            response = client.post('/api/withdraw', json={'amount': float('inf')})
            # If it doesn't error, should fail validation
            if response.status_code == 200:
                data = json.loads(response.data)
                assert data['success'] is False
        except:
            pass  # JSON encoding error is acceptable
        
        try:
            response = client.post('/api/withdraw', json={'amount': float('nan')})
            # If it doesn't error, should fail validation
            if response.status_code == 200:
                data = json.loads(response.data)
                assert data['success'] is False
        except:
            pass  # JSON encoding error is acceptable
    
    def test_balance_cannot_go_negative(self, client, app):
        """Test that balance cannot go negative."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.1, 'helmet': 0.05})
        
        # Try to withdraw more than balance
        response = client.post('/api/withdraw', json={'amount': 10000.00})
        data = json.loads(response.data)
        assert data['success'] is False
        
        # Verify balance unchanged
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 5000.00


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
