"""
API Integration Tests

Tests all Flask API endpoints with valid requests, invalid requests,
security violations, and concurrent requests.
"""

import pytest
import json
import threading
import time
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


class TestIndexEndpoint:
    """Tests for GET / endpoint."""
    
    def test_index_returns_html(self, client):
        """Test that index endpoint returns HTML page."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'ATM Security System' in response.data
        assert b'<html' in response.data


class TestStatusEndpoint:
    """Tests for GET /api/status endpoint."""
    
    def test_status_returns_json(self, client):
        """Test that status endpoint returns JSON."""
        response = client.get('/api/status')
        assert response.status_code == 200
        assert response.content_type == 'application/json'
    
    def test_status_contains_required_fields(self, client):
        """Test that status response contains all required fields."""
        response = client.get('/api/status')
        data = json.loads(response.data)
        
        assert 'screen' in data
        assert 'balance' in data
        assert 'security' in data
        
        security = data['security']
        assert 'access_granted' in security
        assert 'violation_type' in security
        assert 'face_count' in security
        assert 'has_mask' in security
        assert 'has_helmet' in security
        assert 'mask_confidence' in security
        assert 'helmet_confidence' in security
    
    def test_status_initial_state(self, client):
        """Test that initial status is correct."""
        response = client.get('/api/status')
        data = json.loads(response.data)
        
        assert data['screen'] == 'welcome'
        assert data['balance'] == 5000.00


class TestInsertCardEndpoint:
    """Tests for POST /api/insert_card endpoint."""
    
    def test_insert_card_with_access_granted(self, client, app):
        """Test card insertion when access is granted."""
        # Set security status to access granted
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is True
        assert data['screen'] == 'pin_entry'
    
    def test_insert_card_with_access_denied(self, client, app):
        """Test card insertion when access is denied."""
        # Set security status to access denied (mask detected)
        security_manager = app.config['security_manager']
        security_manager.update_status(1, True, False, {'mask': 0.8, 'helmet': 0.0})
        
        response = client.post('/api/insert_card')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'mask' in data['message'].lower()


class TestEnterPinEndpoint:
    """Tests for POST /api/enter_pin endpoint."""
    
    def test_enter_pin_correct(self, client, app):
        """Test PIN entry with correct PIN."""
        # Set up: insert card first
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        client.post('/api/insert_card')
        
        # Enter correct PIN
        response = client.post('/api/enter_pin', 
                              json={'pin': '1234'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is True
        assert data['screen'] == 'menu'
    
    def test_enter_pin_incorrect(self, client, app):
        """Test PIN entry with incorrect PIN."""
        # Set up: insert card first
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        client.post('/api/insert_card')
        
        # Enter incorrect PIN
        response = client.post('/api/enter_pin',
                              json={'pin': '9999'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'Incorrect PIN' in data['message']
    
    def test_enter_pin_with_security_violation(self, client, app):
        """Test PIN entry when security violation occurs."""
        # Set up: insert card first
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        client.post('/api/insert_card')
        
        # Change security status to denied (helmet detected)
        security_manager.update_status(1, False, True, {'mask': 0.0, 'helmet': 0.6})
        
        # Try to enter PIN
        response = client.post('/api/enter_pin',
                              json={'pin': '1234'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()


class TestSelectWithdrawalEndpoint:
    """Tests for POST /api/select_withdrawal endpoint."""
    
    def test_select_withdrawal(self, client):
        """Test selecting withdrawal option."""
        response = client.post('/api/select_withdrawal')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is True
        assert data['screen'] == 'withdrawal'


class TestWithdrawEndpoint:
    """Tests for POST /api/withdraw endpoint."""
    
    def test_withdraw_valid_amount(self, client, app):
        """Test withdrawal with valid amount."""
        # Set up: complete authentication flow
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # Withdraw valid amount
        response = client.post('/api/withdraw',
                              json={'amount': 100.0},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is True
        assert data['new_balance'] == 4900.00
    
    def test_withdraw_zero_amount(self, client, app):
        """Test withdrawal with zero amount."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        response = client.post('/api/withdraw',
                              json={'amount': 0},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'greater than zero' in data['message']
    
    def test_withdraw_exceeds_balance(self, client, app):
        """Test withdrawal exceeding balance."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        response = client.post('/api/withdraw',
                              json={'amount': 10000.0},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'Insufficient balance' in data['message']
    
    def test_withdraw_exceeds_maximum(self, client, app):
        """Test withdrawal exceeding maximum limit."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        response = client.post('/api/withdraw',
                              json={'amount': 1500.0},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'Maximum withdrawal' in data['message']
    
    def test_withdraw_with_security_violation(self, client, app):
        """Test withdrawal when security violation occurs."""
        # Set up: complete authentication flow
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        client.post('/api/insert_card')
        client.post('/api/enter_pin', json={'pin': '1234'})
        client.post('/api/select_withdrawal')
        
        # Change security status to denied
        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Try to withdraw
        response = client.post('/api/withdraw',
                              json={'amount': 100.0},
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'cancelled' in data['message'].lower()


class TestResetEndpoint:
    """Tests for POST /api/reset endpoint."""
    
    def test_reset(self, client):
        """Test resetting ATM to welcome screen."""
        response = client.post('/api/reset')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is True
        assert data['screen'] == 'welcome'


class TestConcurrentRequests:
    """Tests for concurrent API requests."""
    
    def test_concurrent_status_requests(self, client):
        """Test multiple concurrent status requests."""
        results = []
        
        def make_request():
            response = client.get('/api/status')
            results.append(response.status_code)
        
        # Create 10 concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)


class TestCompleteTransactionFlow:
    """Tests for complete transaction flows."""
    
    def test_successful_transaction_flow(self, client, app):
        """Test complete successful transaction from start to finish."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Step 1: Insert card
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # Step 2: Enter PIN
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        assert json.loads(response.data)['success'] is True
        
        # Step 3: Select withdrawal
        response = client.post('/api/select_withdrawal')
        assert json.loads(response.data)['success'] is True
        
        # Step 4: Process withdrawal
        response = client.post('/api/withdraw', json={'amount': 200.0})
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['new_balance'] == 4800.00
        
        # Step 5: Check final status
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['balance'] == 4800.00
    
    def test_transaction_cancelled_by_security_violation(self, client, app):
        """Test transaction cancelled due to security violation."""
        security_manager = app.config['security_manager']
        security_manager.update_status(1, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Step 1: Insert card
        response = client.post('/api/insert_card')
        assert json.loads(response.data)['success'] is True
        
        # Step 2: Security violation occurs (multiple people)
        security_manager.update_status(2, False, False, {'mask': 0.0, 'helmet': 0.0})
        
        # Step 3: Try to enter PIN - should fail
        response = client.post('/api/enter_pin', json={'pin': '1234'})
        data = json.loads(response.data)
        assert data['success'] is False
        
        # Step 4: Check ATM reset to welcome
        response = client.get('/api/status')
        data = json.loads(response.data)
        assert data['screen'] == 'welcome'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
