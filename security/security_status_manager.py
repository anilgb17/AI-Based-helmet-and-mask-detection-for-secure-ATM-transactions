"""
Security Status Manager Module

Manages security state and access control decisions based on face detection,
mask detection, and helmet detection results.
"""

import time
import logging

logger = logging.getLogger(__name__)


class SecurityStatusManager:
    """
    Centralized management of security state and access control decisions.
    
    Attributes:
        access_granted (bool): Whether user can proceed with ATM operations
        violation_type (str|None): Description of current violation
        face_count (int): Number of faces detected in current frame
        has_mask (bool): Whether mask is detected
        has_helmet (bool): Whether helmet is detected
        mask_confidence (float): Confidence score for mask detection
        helmet_confidence (float): Confidence score for helmet detection
        warning_count (int): Counter for consecutive violations
    """
    
    def __init__(self):
        """Initialize security status manager with default values."""
        self.access_granted = False
        self.violation_type = "no_face_detected"
        self.face_count = 0
        self.has_mask = False
        self.has_helmet = False
        self.mask_confidence = 0.0
        self.helmet_confidence = 0.0
        self.warning_count = 0
        self.timestamp = time.time()
        logger.info("Security status manager initialized")
    
    def update_status(self, face_count, has_mask, has_helmet, confidences, is_multiple_persons=False):
        """
        Update security state based on detection results.
        
        Args:
            face_count (int): Number of faces detected
            has_mask (bool): Whether mask is detected
            has_helmet (bool): Whether helmet is detected
            confidences (dict): Dictionary with 'mask', 'helmet', and 'person' confidence scores
            is_multiple_persons (bool): Whether multiple persons detected by CNN model
        """
        # Store previous state for change detection
        previous_access = self.access_granted
        previous_violation = self.violation_type
        
        self.face_count = face_count
        self.has_mask = has_mask
        self.has_helmet = has_helmet
        self.mask_confidence = confidences.get('mask', 0.0)
        self.helmet_confidence = confidences.get('helmet', 0.0)
        self.person_confidence = confidences.get('person', 0.0)
        self.is_multiple_persons = is_multiple_persons
        self.timestamp = time.time()
        
        # IMPROVED LOGIC: Better multiple people detection
        # Trigger multiple_people if ANY of these conditions:
        # 1. Face count >= 2 (2 or more faces detected)
        # 2. Person detector detects multiple persons with confidence > 0.3
        
        # Apply security logic based on requirements
        if face_count == 0:
            self.access_granted = False
            self.violation_type = "no_face_detected"
            self.warning_count += 1
        elif face_count >= 2 or (is_multiple_persons and self.person_confidence > 0.3):
            # Deny if 2+ faces OR person detector says multiple with reasonable confidence
            self.access_granted = False
            self.violation_type = "multiple_people"
            self.warning_count += 1
        elif has_mask and has_helmet:
            self.access_granted = False
            self.violation_type = "mask_and_helmet_detected"
            self.warning_count += 1
        elif has_mask:
            self.access_granted = False
            self.violation_type = "mask_detected"
            self.warning_count += 1
        elif has_helmet:
            self.access_granted = False
            self.violation_type = "helmet_detected"
            self.warning_count += 1
        else:
            self.access_granted = True
            self.violation_type = None
            self.reset_warnings()
        
        # Log status changes
        if previous_access != self.access_granted or previous_violation != self.violation_type:
            if self.access_granted:
                logger.info(f"Security status: ACCESS GRANTED (faces={face_count})")
            else:
                logger.warning(
                    f"Security status: ACCESS DENIED - {self.violation_type} "
                    f"(faces={face_count}, mask={has_mask}:{self.mask_confidence:.2f}, "
                    f"helmet={has_helmet}:{self.helmet_confidence:.2f}, warnings={self.warning_count})"
                )
        
        # Log persistent violations
        if self.warning_count > 0 and self.warning_count % 50 == 0:
            logger.warning(f"Persistent security violation: {self.warning_count} consecutive warnings")
    
    def get_status(self):
        """
        Return current status as dictionary.
        
        Returns:
            dict: Complete security status information
        """
        return {
            'access_granted': self.access_granted,
            'violation_type': self.violation_type,
            'face_count': self.face_count,
            'has_mask': self.has_mask,
            'has_helmet': self.has_helmet,
            'mask_confidence': self.mask_confidence,
            'helmet_confidence': self.helmet_confidence,
            'warning_count': self.warning_count,
            'timestamp': self.timestamp
        }
    
    def reset_warnings(self):
        """Reset the warning counter to zero."""
        self.warning_count = 0
    
    def get_violation_message(self):
        """
        Return user-friendly violation description.
        
        Returns:
            str: Human-readable message describing the violation, or None if access granted
        """
        if self.violation_type is None:
            return None
        
        messages = {
            'no_face_detected': 'No face detected. Please position yourself in front of the camera.',
            'multiple_people': 'Multiple people detected. Only one person allowed at the ATM.',
            'mask_detected': 'Face mask detected. Please remove your mask to proceed.',
            'helmet_detected': 'Helmet detected. Please remove your helmet to proceed.',
            'mask_and_helmet_detected': 'Mask and helmet detected. Please remove face coverings to proceed.'
        }
        
        return messages.get(self.violation_type, 'Security violation detected.')
