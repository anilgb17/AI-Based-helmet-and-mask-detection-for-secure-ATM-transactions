"""
Integration tests for Transaction Controller
"""

import unittest
from atm.atm_state_manager import ATMStateManager
from atm.transaction_controller import TransactionController
from security.security_status_manager import SecurityStatusManager


class TestTransactionController(unittest.TestCase):
    """Integration test cases for TransactionController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atm_state = ATMStateManager(initial_balance=5000.00)
        self.security_status = SecurityStatusManager()
        self.controller = TransactionController(self.atm_state, self.security_status)
    
    def test_complete_transaction_flow_with_access_granted(self):
        """Test complete transaction flow when access is granted throughout."""
        # Set security status to access granted (single face, no mask, no helmet)
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        
        # Step 1: Insert card
        result = self.controller.handle_card_insertion()
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'pin_entry')
        self.assertIn('Card inserted successfully', result['message'])
        
        # Step 2: Enter PIN
        result = self.controller.handle_pin_entry('1234')
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'menu')
        self.assertIn('PIN accepted', result['message'])
        
        # Step 3: Select withdrawal
        result = self.atm_state.select_withdrawal()
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'withdrawal')
        
        # Step 4: Process withdrawal
        result = self.controller.handle_withdrawal(200.00)
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'complete')
        self.assertEqual(result['new_balance'], 4800.00)
        self.assertEqual(self.atm_state.balance, 4800.00)
    
    def test_card_insertion_with_mask_violation(self):
        """Test card insertion fails when mask is detected."""
        # Set security status to mask detected
        self.security_status.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.7, 'helmet': 0.1}
        )
        
        result = self.controller.handle_card_insertion()
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('mask', result['message'].lower())
    
    def test_card_insertion_with_helmet_violation(self):
        """Test card insertion fails when helmet is detected."""
        # Set security status to helmet detected
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=True,
            confidences={'mask': 0.1, 'helmet': 0.6}
        )
        
        result = self.controller.handle_card_insertion()
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('helmet', result['message'].lower())
    
    def test_card_insertion_with_multiple_people(self):
        """Test card insertion fails when multiple people are detected."""
        # Set security status to multiple people
        self.security_status.update_status(
            face_count=2,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        result = self.controller.handle_card_insertion()
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('multiple', result['message'].lower())
    
    def test_card_insertion_with_no_face(self):
        """Test card insertion fails when no face is detected."""
        # Set security status to no face detected
        self.security_status.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        
        result = self.controller.handle_card_insertion()
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('face', result['message'].lower())
    
    def test_pin_entry_security_violation_cancels_transaction(self):
        """Test that security violation during PIN entry cancels transaction."""
        # Start with access granted
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        
        # Insert card successfully
        result = self.controller.handle_card_insertion()
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'pin_entry')
        
        # Security violation occurs (mask detected)
        self.security_status.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.8, 'helmet': 0.1}
        )
        
        # Try to enter PIN
        result = self.controller.handle_pin_entry('1234')
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('cancelled', result['message'].lower())
    
    def test_withdrawal_security_violation_cancels_transaction(self):
        """Test that security violation during withdrawal cancels transaction."""
        # Start with access granted
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        
        # Complete card insertion and PIN entry
        self.controller.handle_card_insertion()
        self.controller.handle_pin_entry('1234')
        self.atm_state.select_withdrawal()
        
        # Security violation occurs (helmet detected)
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=True,
            confidences={'mask': 0.1, 'helmet': 0.7}
        )
        
        # Try to process withdrawal
        result = self.controller.handle_withdrawal(100.00)
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('cancelled', result['message'].lower())
        
        # Verify balance unchanged
        self.assertEqual(self.atm_state.balance, 5000.00)
    
    def test_security_validation_at_each_step(self):
        """Test that security is validated at each transaction step."""
        # Step 1: Card insertion with no face - should fail
        self.security_status.update_status(
            face_count=0,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        result = self.controller.handle_card_insertion()
        self.assertFalse(result['success'])
        
        # Step 2: Grant access and insert card
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        result = self.controller.handle_card_insertion()
        self.assertTrue(result['success'])
        
        # Step 3: PIN entry with multiple people - should fail and cancel
        self.security_status.update_status(
            face_count=2,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.0, 'helmet': 0.0}
        )
        result = self.controller.handle_pin_entry('1234')
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        
        # Step 4: Start over with access granted
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        self.controller.handle_card_insertion()
        self.controller.handle_pin_entry('1234')
        self.atm_state.select_withdrawal()
        
        # Step 5: Withdrawal with mask - should fail and cancel
        self.security_status.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.9, 'helmet': 0.1}
        )
        result = self.controller.handle_withdrawal(50.00)
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertEqual(self.atm_state.balance, 5000.00)
    
    def test_validate_security_method(self):
        """Test the validate_security helper method."""
        # Test with access granted
        status_granted = {'access_granted': True}
        self.assertTrue(self.controller.validate_security(status_granted))
        
        # Test with access denied
        status_denied = {'access_granted': False}
        self.assertFalse(self.controller.validate_security(status_denied))
        
        # Test with missing key (should default to False)
        status_missing = {}
        self.assertFalse(self.controller.validate_security(status_missing))
    
    def test_cancel_transaction_method(self):
        """Test the cancel_transaction method."""
        # Set up a transaction in progress
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        self.controller.handle_card_insertion()
        self.controller.handle_pin_entry('1234')
        
        # Cancel transaction
        result = self.controller.cancel_transaction('Test cancellation')
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('cancelled', result['message'].lower())
        self.assertIn('Test cancellation', result['message'])
        
        # Verify ATM is reset
        self.assertEqual(self.atm_state.current_screen, 'welcome')
        self.assertFalse(self.atm_state.card_inserted)
    
    def test_multiple_transaction_attempts(self):
        """Test multiple transaction attempts with varying security status."""
        # First attempt - successful
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        self.controller.handle_card_insertion()
        self.controller.handle_pin_entry('1234')
        self.atm_state.select_withdrawal()
        result = self.controller.handle_withdrawal(100.00)
        self.assertTrue(result['success'])
        self.assertEqual(self.atm_state.balance, 4900.00)
        
        # Reset for second attempt
        self.atm_state.reset()
        
        # Second attempt - fails at card insertion
        self.security_status.update_status(
            face_count=1,
            has_mask=True,
            has_helmet=False,
            confidences={'mask': 0.8, 'helmet': 0.1}
        )
        result = self.controller.handle_card_insertion()
        self.assertFalse(result['success'])
        
        # Third attempt - successful
        self.security_status.update_status(
            face_count=1,
            has_mask=False,
            has_helmet=False,
            confidences={'mask': 0.1, 'helmet': 0.05}
        )
        self.controller.handle_card_insertion()
        self.controller.handle_pin_entry('1234')
        self.atm_state.select_withdrawal()
        result = self.controller.handle_withdrawal(50.00)
        self.assertTrue(result['success'])
        self.assertEqual(self.atm_state.balance, 4850.00)


if __name__ == '__main__':
    unittest.main()
