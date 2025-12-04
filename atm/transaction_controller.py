"""
Transaction Controller Module

Coordinates between security monitoring and ATM state management to ensure
all transaction steps validate security status before proceeding.
"""

import logging

logger = logging.getLogger(__name__)


class TransactionController:
    """
    Coordinates security validation with ATM transaction processing.
    
    Ensures that every transaction step validates current security status
    before allowing the operation to proceed.
    """
    
    def __init__(self, atm_state_manager, security_status_manager):
        """
        Initialize transaction controller.
        
        Args:
            atm_state_manager: ATMStateManager instance
            security_status_manager: SecurityStatusManager instance
        """
        self.atm_state = atm_state_manager
        self.security_status = security_status_manager
        logger.info("Transaction controller initialized")
    
    def handle_card_insertion(self):
        """
        Process card insertion request with security validation.
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        logger.info("Handling card insertion request")
        
        try:
            # Get current security status
            security_status = self.security_status.get_status()
            
            # Validate security before proceeding
            if not self.validate_security(security_status):
                logger.warning(f"Card insertion blocked: {self.security_status.get_violation_message()}")
                return {
                    'success': False,
                    'message': self.security_status.get_violation_message(),
                    'screen': self.atm_state.current_screen
                }
            
            # Process card insertion through ATM state manager
            result = self.atm_state.insert_card(security_status)
            logger.info(f"Card insertion result: {result['success']}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling card insertion: {e}")
            return {
                'success': False,
                'message': 'System error during card insertion',
                'screen': self.atm_state.current_screen
            }
    
    def handle_pin_entry(self, pin):
        """
        Process PIN entry with security validation.
        
        Args:
            pin: 4-digit PIN string
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        logger.info("Handling PIN entry request")
        
        try:
            # Get current security status
            security_status = self.security_status.get_status()
            
            # Validate security before proceeding
            if not self.validate_security(security_status):
                logger.warning("PIN entry blocked due to security violation")
                return self.cancel_transaction('Security violation during PIN entry')
            
            # Process PIN entry through ATM state manager
            result = self.atm_state.enter_pin(pin, security_status)
            logger.info(f"PIN entry result: {result['success']}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling PIN entry: {e}")
            return {
                'success': False,
                'message': 'System error during PIN entry',
                'screen': self.atm_state.current_screen
            }
    
    def handle_withdrawal(self, amount):
        """
        Process withdrawal request with security validation.
        
        Args:
            amount: Withdrawal amount (float)
        
        Returns:
            Dictionary with 'success', 'message', 'screen', and optionally 'new_balance'
        """
        logger.info(f"Handling withdrawal request: ${amount}")
        
        try:
            # Get current security status
            security_status = self.security_status.get_status()
            
            # Validate security before proceeding
            if not self.validate_security(security_status):
                logger.warning("Withdrawal blocked due to security violation")
                return self.cancel_transaction('Security violation during withdrawal')
            
            # Process withdrawal through ATM state manager
            result = self.atm_state.process_withdrawal(amount, security_status)
            logger.info(f"Withdrawal result: {result['success']}, amount: ${amount}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling withdrawal: {e}")
            return {
                'success': False,
                'message': 'System error during withdrawal',
                'screen': self.atm_state.current_screen
            }
    
    def validate_security(self, security_status):
        """
        Check if transaction can proceed based on security status.
        
        Args:
            security_status: Dictionary with security status information
        
        Returns:
            bool: True if access is granted, False otherwise
        """
        return security_status.get('access_granted', False)
    
    def cancel_transaction(self, reason):
        """
        Abort transaction and reset ATM to welcome screen.
        
        Args:
            reason: String describing why transaction was cancelled
        
        Returns:
            Dictionary with 'success', 'message', and 'screen'
        """
        logger.warning(f"Transaction cancelled: {reason}")
        
        try:
            # Reset ATM state
            self.atm_state.reset()
            
            return {
                'success': False,
                'message': f'Transaction cancelled: {reason}',
                'screen': self.atm_state.current_screen
            }
            
        except Exception as e:
            logger.error(f"Error during transaction cancellation: {e}")
            return {
                'success': False,
                'message': 'Transaction cancelled',
                'screen': 'welcome'  # Fallback to welcome screen
            }
