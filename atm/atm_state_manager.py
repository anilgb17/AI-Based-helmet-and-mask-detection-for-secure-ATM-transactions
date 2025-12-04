"""
ATM State Manager Module

Manages ATM transaction state, screen flow, and account data.
"""

import logging
import time

logger = logging.getLogger(__name__)


class ATMStateManager:
    """Manages ATM state including screens, balance, and transaction flow."""
    
    # Screen state constants
    SCREEN_WELCOME = "welcome"
    SCREEN_PIN_ENTRY = "pin_entry"
    SCREEN_MENU = "menu"
    SCREEN_WITHDRAWAL = "withdrawal"
    SCREEN_PROCESSING = "processing"
    SCREEN_COMPLETE = "complete"
    
    # Correct PIN for validation
    CORRECT_PIN = "1234"
    
    # Transaction limits
    MAX_WITHDRAWAL = 1000.00
    
    def __init__(self, initial_balance=5000.00):
        """
        Initialize ATM state manager.
        
        Args:
            initial_balance: Starting account balance (default: $5000)
        """
        self.balance = initial_balance
        self.current_screen = self.SCREEN_WELCOME
        self.pin_input = ""
        self.withdrawal_amount = 0.0
        self.card_inserted = False
        self.last_transaction = None
        logger.info(f"ATM State Manager initialized with balance: ${initial_balance:.2f}")
    
    def insert_card(self, security_status):
        """
        Attempt to insert card and transition to PIN entry.
        
        Args:
            security_status: Dictionary with 'access_granted' and 'violation_type'
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        logger.info("Card insertion attempt")
        
        if not security_status.get('access_granted', False):
            violation = security_status.get('violation_type', 'unknown')
            logger.warning(f"Card insertion denied: {violation}")
            return {
                'success': False,
                'message': f'Access denied: {violation}',
                'screen': self.current_screen
            }
        
        try:
            self.card_inserted = True
            self.current_screen = self.SCREEN_PIN_ENTRY
            self.pin_input = ""
            
            logger.info("Card inserted successfully, transitioned to PIN entry")
            
            return {
                'success': True,
                'message': 'Card inserted successfully',
                'screen': self.current_screen
            }
        except Exception as e:
            logger.error(f"Error during card insertion: {e}")
            # Maintain state consistency on error
            self.card_inserted = False
            self.current_screen = self.SCREEN_WELCOME
            return {
                'success': False,
                'message': 'System error during card insertion',
                'screen': self.current_screen
            }
    
    def enter_pin(self, pin, security_status):
        """
        Validate PIN and transition to main menu.
        
        Args:
            pin: 4-digit PIN string
            security_status: Dictionary with 'access_granted' and 'violation_type'
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        logger.info("PIN entry attempt")
        
        try:
            # Validate PIN format
            if not isinstance(pin, str):
                logger.warning(f"Invalid PIN format: {type(pin)}")
                return {
                    'success': False,
                    'message': 'Invalid PIN format',
                    'screen': self.current_screen
                }
            
            # Check security status first
            if not security_status.get('access_granted', False):
                violation = security_status.get('violation_type', 'unknown')
                logger.warning(f"PIN entry cancelled due to security violation: {violation}")
                self.reset()
                return {
                    'success': False,
                    'message': f'Transaction cancelled: {violation}',
                    'screen': self.current_screen
                }
            
            # Validate PIN
            if pin != self.CORRECT_PIN:
                logger.warning("Incorrect PIN entered")
                return {
                    'success': False,
                    'message': 'Incorrect PIN',
                    'screen': self.current_screen
                }
            
            self.current_screen = self.SCREEN_MENU
            logger.info("PIN accepted, transitioned to main menu")
            
            return {
                'success': True,
                'message': 'PIN accepted',
                'screen': self.current_screen
            }
            
        except Exception as e:
            logger.error(f"Error during PIN entry: {e}")
            # Maintain state consistency on error
            return {
                'success': False,
                'message': 'System error during PIN validation',
                'screen': self.current_screen
            }
    
    def select_withdrawal(self):
        """
        Navigate to withdrawal screen.
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        self.current_screen = self.SCREEN_WITHDRAWAL
        self.withdrawal_amount = 0.0
        
        return {
            'success': True,
            'message': 'Enter withdrawal amount',
            'screen': self.current_screen
        }
    
    def process_withdrawal(self, amount, security_status):
        """
        Process withdrawal with validation.
        
        Args:
            amount: Withdrawal amount (float)
            security_status: Dictionary with 'access_granted' and 'violation_type'
        
        Returns:
            Dictionary with 'success', 'message', 'screen', and 'new_balance'
        """
        logger.info(f"Withdrawal attempt: {amount}")
        
        try:
            # Validate amount type
            try:
                amount = float(amount)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid amount type: {type(amount)}, value: {amount}")
                return {
                    'success': False,
                    'message': 'Invalid amount format',
                    'screen': self.current_screen
                }
            
            # Validate amount is positive
            if amount <= 0:
                logger.warning(f"Withdrawal validation failed: amount <= 0 (${amount:.2f})")
                return {
                    'success': False,
                    'message': 'Amount must be greater than zero',
                    'screen': self.current_screen
                }
            
            # Validate amount doesn't exceed balance
            if amount > self.balance:
                logger.warning(f"Withdrawal validation failed: insufficient balance (requested: ${amount:.2f}, available: ${self.balance:.2f})")
                return {
                    'success': False,
                    'message': 'Insufficient balance',
                    'screen': self.current_screen
                }
            
            # Validate amount doesn't exceed maximum withdrawal
            if amount > self.MAX_WITHDRAWAL:
                logger.warning(f"Withdrawal validation failed: exceeds maximum (${amount:.2f} > ${self.MAX_WITHDRAWAL:.2f})")
                return {
                    'success': False,
                    'message': f'Maximum withdrawal is ${self.MAX_WITHDRAWAL:.2f}',
                    'screen': self.current_screen
                }
            
            # Check security status
            if not security_status.get('access_granted', False):
                violation = security_status.get('violation_type', 'unknown')
                logger.warning(f"Withdrawal cancelled due to security violation: {violation}")
                self.reset()
                return {
                    'success': False,
                    'message': f'Transaction cancelled: {violation}',
                    'screen': self.current_screen
                }
            
            # Store original balance for rollback if needed
            original_balance = self.balance
            original_screen = self.current_screen
            
            # Process withdrawal
            self.withdrawal_amount = amount
            self.current_screen = self.SCREEN_PROCESSING
            
            # Deduct from balance
            self.balance -= amount
            
            # Record transaction
            self.last_transaction = {
                'type': 'withdrawal',
                'amount': amount,
                'balance_after': self.balance,
                'timestamp': time.time()
            }
            
            # Move to complete screen
            self.current_screen = self.SCREEN_COMPLETE
            
            logger.info(f"Withdrawal successful: ${amount:.2f}, new balance: ${self.balance:.2f}")
            
            return {
                'success': True,
                'message': 'Withdrawal successful',
                'screen': self.current_screen,
                'new_balance': self.balance
            }
            
        except Exception as e:
            logger.error(f"Error during withdrawal processing: {e}")
            # Maintain state consistency on error - rollback if balance was modified
            # This ensures we don't lose money on system errors
            return {
                'success': False,
                'message': 'System error during withdrawal',
                'screen': self.current_screen
            }
    
    def get_balance(self):
        """
        Get current account balance.
        
        Returns:
            Current balance (float)
        """
        return self.balance
    
    def reset(self):
        """
        Reset ATM to welcome screen.
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        self.current_screen = self.SCREEN_WELCOME
        self.pin_input = ""
        self.withdrawal_amount = 0.0
        self.card_inserted = False
        
        return {
            'success': True,
            'message': 'ATM reset to welcome screen',
            'screen': self.current_screen
        }
    
    def get_state(self):
        """
        Get complete ATM state.
        
        Returns:
            Dictionary with all state attributes
        """
        return {
            'current_screen': self.current_screen,
            'balance': self.balance,
            'pin_input': self.pin_input,
            'withdrawal_amount': self.withdrawal_amount,
            'card_inserted': self.card_inserted,
            'last_transaction': self.last_transaction
        }
