"""
Frontend Integration Tests

Tests frontend HTML/CSS/JavaScript integration with backend APIs,
screen transitions, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
import sys
import os

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
        mock_cam_instance.read.return_value = (True, Mock())
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


class TestHTMLRendering:
    """Tests for HTML page rendering."""
    
    def test_index_page_loads(self, client):
        """Test that index page loads successfully."""
        response = client.get('/')
        assert response.status_code == 200
        assert response.content_type == 'text/html; charset=utf-8'
    
    def test_index_contains_required_elements(self, client):
        """Test that index page contains all required UI elements."""
        response = client.get('/')
        html = response.data.decode('utf-8')
        
        # Check for main sections
        assert 'video-section' in html
        assert 'atm-section' in html
        assert 'security-status' in html
        
        # Check for all screens
        assert 'welcome-screen' in html
        assert 'pin-screen' in html
        assert 'menu-screen' in html
        assert 'withdrawal-screen' in html
        assert 'processing-screen' in html
        assert 'complete-screen' in html
    
    def test_index_includes_css(self, client):
        """Test that index page includes CSS stylesheet."""
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'style.css' in html
    
    def test_index_includes_javascript(self, client):
        """Test that index page includes JavaScript file."""
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'script.js' in html


class TestScreenTransitions:
    """Tests for screen transition flows."""
    
    def test_welcome_to_pin_transition(self, client, app):
        """Test transition from welcome to PIN entry screen."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Insert card
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert data['screen'] == 'pin_entry'
        
        # Verify status reflects new screen
        status_response = client.get('/api/status')
        status_data = json.loads(status_response.data)
        assert status_data['screen'] == 'pin_entry'
    
    def test_pin_to_menu_transition(self, client, app):
        """Test transition from PIN entry to main menu."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Complete card insertion and PIN entry
        client.post('/api/insert_card')
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert data['screen'] == 'menu'
    
    def test_menu_to_withdrawal_transition(self, client, app):
        """Test transition from menu to withdrawal screen."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Complete authentication
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        
        # Select withdrawal
        response = client.post('/api/select_withdrawal')
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert data['screen'] == 'withdrawal'
    
    def test_withdrawal_to_complete_transition(self, client, app):
        """Test transition from withdrawal to completion screen."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Complete full flow
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # Process withdrawal
        response = client.post('/api/withdraw', json={'amount': 50.0})
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'new_balance' in data
    
    def test_reset_to_welcome_transition(self, client, app):
        """Test reset returns to welcome screen from any state."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Navigate to menu
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        
        # Reset
        response = client.post('/api/reset')
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert data['screen'] == 'welcome'


class TestAPIErrorHandling:
    """Tests for API error handling in frontend integration."""
    
    def test_card_insertion_denied_error(self, client, app):
        """Test error handling when card insertion is denied."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.0})
        
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        
        assert data['success'] is False
        assert 'message' in data
        assert len(data['message']) > 0
    
    def test_incorrect_pin_error(self, client, app):
        """Test error handling for incorrect PIN."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        client.post('/api/insert_card')
        response = client.post('/api/enter_pin', json={'pin': '0000'})
        data = json.loads(response.data)
        
        assert data['success'] is False
        assert 'Incorrect PIN' in data['message']
    
    def test_invalid_withdrawal_amount_error(self, client, app):
        """Test error handling for invalid withdrawal amounts."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Test zero amount
        response = client.post('/api/withdraw', json={'amount': 0})
        data = json.loads(response.data)
        assert data['success'] is False
        
        # Test negative amount
        response = client.post('/api/withdraw', json={'amount': -100})
        data = json.loads(response.data)
        assert data['success'] is False
        
        # Test amount exceeding maximum
        response = client.post('/api/withdraw', json={'amount': 2000})
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_insufficient_balance_error(self, client, app):
        """Test error handling for insufficient balance."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        response = client.post('/api/withdraw', json={'amount': 10000})
        data = json.loads(response.data)
        
        assert data['success'] is False
        assert 'Insufficient balance' in data['message']


class TestStatusPolling:
    """Tests for status polling functionality."""
    
    def test_status_endpoint_response_time(self, client):
        """Test that status endpoint responds quickly for polling."""
        import time
        
        start_time = time.time()
        response = client.get('/api/status')
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        # Should respond in less than 100ms as per requirements
        assert response_time < 100
    
    def test_status_updates_balance(self, client, app):
        """Test that status endpoint reflects balance changes."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Get initial balance
        response = client.get('/api/status')
        initial_data = json.loads(response.data)
        initial_balance = initial_data['balance']
        
        # Perform withdrawal
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        client.post('/api/withdraw', json={'amount': 100})
        
        # Check updated balance
        response = client.get('/api/status')
        updated_data = json.loads(response.data)
        
        assert updated_data['balance'] == initial_balance - 100
    
    def test_status_updates_security_state(self, client, app):
        """Test that status endpoint reflects security state changes."""
        security_manager = app.config['security_manager']
        
        # Set access granted
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is True
        
        # Change to access denied
        security_manager.update_status(1, True, False, {'mask': 0.7, 'helmet': 0.0})
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['security']['access_granted'] is False
        assert data['security']['violation_type'] == 'mask_detected'


class TestCompleteUserFlows:
    """Tests for complete user interaction flows."""
    
    def test_successful_withdrawal_flow(self, client, app):
        """Test complete successful withdrawal flow."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Step 1: Check initial state
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['screen'] == 'welcome'
        initial_balance = data['balance']
        
        # Step 2: Insert card
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # Step 3: Enter PIN
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        assert json.loads(response.data)['success'] is True
        
        # Step 4: Select withdrawal
        response = client.post('/api/select_withdrawal')
        assert json.loads(response.data)['success'] is True
        
        # Step 5: Process withdrawal
        withdrawal_amount = 150.0
        response = client.post('/api/withdraw', json={'amount': withdrawal_amount})
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['new_balance'] == initial_balance - withdrawal_amount
        
        # Step 6: Verify final state
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == initial_balance - withdrawal_amount
    
    def test_cancelled_transaction_flow(self, client, app):
        """Test transaction cancelled by user."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Start transaction
        client.post('/api/insert_card')
        
        # Cancel during PIN entry
        response = client.post('/api/reset')
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert data['screen'] == 'welcome'
        
        # Verify balance unchanged
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 5000.00
    
    def test_security_violation_during_transaction(self, client, app):
        """Test transaction interrupted by security violation."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Start transaction
        client.post('/api/insert_card')
        
        # Security violation occurs
        security_manager.update_status(1, False, True, {'mask': 0.0, 'helmet': 0.5})
        
        # Try to continue - should fail
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
