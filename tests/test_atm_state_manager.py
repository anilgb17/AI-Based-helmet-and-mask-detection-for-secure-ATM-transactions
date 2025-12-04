"""
Unit tests for ATM State Manager
"""

import unittest
from atm.atm_state_manager import ATMStateManager


class TestATMStateManager(unittest.TestCase):
    """Test cases for ATMStateManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atm = ATMStateManager(initial_balance=5000.00)
        self.security_granted = {'access_granted': True, 'violation_type': None}
        self.security_denied = {'access_granted': False, 'violation_type': 'mask_detected'}
    
    def test_initial_state(self):
        """Test initial ATM state."""
        state = self.atm.get_state()
        self.assertEqual(state['current_screen'], 'welcome')
        self.assertEqual(state['balance'], 5000.00)
        self.assertEqual(state['pin_input'], '')
        self.assertEqual(state['withdrawal_amount'], 0.0)
        self.assertFalse(state['card_inserted'])
        self.assertIsNone(state['last_transaction'])
    
    def test_insert_card_success(self):
        """Test successful card insertion with access granted."""
        result = self.atm.insert_card(self.security_granted)
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'pin_entry')
        self.assertIn('Card inserted successfully', result['message'])
        self.assertTrue(self.atm.card_inserted)
    
    def test_insert_card_denied(self):
        """Test card insertion with access denied."""
        result = self.atm.insert_card(self.security_denied)
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('Access denied', result['message'])
        self.assertFalse(self.atm.card_inserted)
    
    def test_enter_pin_correct(self):
        """Test PIN entry with correct PIN."""
        self.atm.insert_card(self.security_granted)
        result = self.atm.enter_pin('1234', self.security_granted)
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'menu')
        self.assertIn('PIN accepted', result['message'])
    
    def test_enter_pin_incorrect(self):
        """Test PIN entry with incorrect PIN."""
        self.atm.insert_card(self.security_granted)
        result = self.atm.enter_pin('9999', self.security_granted)
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'pin_entry')
        self.assertIn('Incorrect PIN', result['message'])
    
    def test_enter_pin_security_violation(self):
        """Test PIN entry with security violation."""
        self.atm.insert_card(self.security_granted)
        result = self.atm.enter_pin('1234', self.security_denied)
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('Transaction cancelled', result['message'])
    
    def test_select_withdrawal(self):
        """Test navigation to withdrawal screen."""
        result = self.atm.select_withdrawal()
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'withdrawal')
        self.assertEqual(self.atm.withdrawal_amount, 0.0)
    
    def test_process_withdrawal_success(self):
        """Test successful withdrawal."""
        result = self.atm.process_withdrawal(100.00, self.security_granted)
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'complete')
        self.assertEqual(result['new_balance'], 4900.00)
        self.assertEqual(self.atm.balance, 4900.00)
        self.assertIn('Withdrawal successful', result['message'])
    
    def test_process_withdrawal_zero_amount(self):
        """Test withdrawal with zero amount."""
        result = self.atm.process_withdrawal(0.0, self.security_granted)
        self.assertFalse(result['success'])
        self.assertIn('greater than zero', result['message'])
        self.assertEqual(self.atm.balance, 5000.00)
    
    def test_process_withdrawal_negative_amount(self):
        """Test withdrawal with negative amount."""
        result = self.atm.process_withdrawal(-50.00, self.security_granted)
        self.assertFalse(result['success'])
        self.assertIn('greater than zero', result['message'])
        self.assertEqual(self.atm.balance, 5000.00)
    
    def test_process_withdrawal_exceeds_balance(self):
        """Test withdrawal exceeding balance."""
        result = self.atm.process_withdrawal(6000.00, self.security_granted)
        self.assertFalse(result['success'])
        self.assertIn('Insufficient balance', result['message'])
        self.assertEqual(self.atm.balance, 5000.00)
    
    def test_process_withdrawal_exceeds_max_limit(self):
        """Test withdrawal exceeding maximum limit."""
        result = self.atm.process_withdrawal(1500.00, self.security_granted)
        self.assertFalse(result['success'])
        self.assertIn('Maximum withdrawal', result['message'])
        self.assertEqual(self.atm.balance, 5000.00)
    
    def test_process_withdrawal_security_violation(self):
        """Test withdrawal with security violation."""
        result = self.atm.process_withdrawal(100.00, self.security_denied)
        self.assertFalse(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        self.assertIn('Transaction cancelled', result['message'])
        self.assertEqual(self.atm.balance, 5000.00)
    
    def test_get_balance(self):
        """Test getting current balance."""
        balance = self.atm.get_balance()
        self.assertEqual(balance, 5000.00)
        
        # After withdrawal
        self.atm.process_withdrawal(200.00, self.security_granted)
        balance = self.atm.get_balance()
        self.assertEqual(balance, 4800.00)
    
    def test_reset(self):
        """Test ATM reset to welcome screen."""
        # Perform some operations
        self.atm.insert_card(self.security_granted)
        self.atm.enter_pin('1234', self.security_granted)
        
        # Reset
        result = self.atm.reset()
        self.assertTrue(result['success'])
        self.assertEqual(result['screen'], 'welcome')
        
        # Verify state is reset
        state = self.atm.get_state()
        self.assertEqual(state['current_screen'], 'welcome')
        self.assertEqual(state['pin_input'], '')
        self.assertEqual(state['withdrawal_amount'], 0.0)
        self.assertFalse(state['card_inserted'])
    
    def test_complete_transaction_flow(self):
        """Test complete transaction flow from start to finish."""
        # Insert card
        result = self.atm.insert_card(self.security_granted)
        self.assertTrue(result['success'])
        self.assertEqual(self.atm.current_screen, 'pin_entry')
        
        # Enter PIN
        result = self.atm.enter_pin('1234', self.security_granted)
        self.assertTrue(result['success'])
        self.assertEqual(self.atm.current_screen, 'menu')
        
        # Select withdrawal
        result = self.atm.select_withdrawal()
        self.assertTrue(result['success'])
        self.assertEqual(self.atm.current_screen, 'withdrawal')
        
        # Process withdrawal
        result = self.atm.process_withdrawal(250.00, self.security_granted)
        self.assertTrue(result['success'])
        self.assertEqual(self.atm.current_screen, 'complete')
        self.assertEqual(self.atm.balance, 4750.00)
        
        # Verify transaction was recorded
        self.assertIsNotNone(self.atm.last_transaction)
        self.assertEqual(self.atm.last_transaction['type'], 'withdrawal')
        self.assertEqual(self.atm.last_transaction['amount'], 250.00)
        self.assertEqual(self.atm.last_transaction['balance_after'], 4750.00)


if __name__ == '__main__':
    unittest.main()
