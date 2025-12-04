"""
================================================================================
ATM SECURITY SYSTEM - COMPLETE INTEGRATED APPLICATION
================================================================================

SYSTEM OVERVIEW:
This is a comprehensive ATM Security System that uses multiple AI/ML models
to ensure secure transactions by detecting and preventing unauthorized access.

MAIN COMPONENTS:
1. Face Detection (Haar Cascade)
2. Face Mask Detection (PyTorch CNN)
3. Helmet Detection (PyTorch CNN)
4. Person Count Detection (PyTorch CNN)
5. Security Violation Management
6. ATM Transaction Processing
7. Video Streaming with Real-time Overlays
8. Audio Alert System

SECURITY LOGIC:
- GRANT ACCESS: Single person, clear face (no mask, no helmet)
- DENY ACCESS: Multiple people, masked face, helmeted person, no face detected

Author: ATM Security System Development Team
Version: 3.0 (Complete Integrated)
Date: November 19, 2025
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORT STATEMENTS
# ============================================================================

# Standard library imports
import json
import logging
import sys
import time
import signal

# Third-party imports
import cv2
from flask import Flask, Response, jsonify, request, render_template

# Logging configuration
from logging.handlers import RotatingFileHandler


# ============================================================================
# SECTION 2: AI/ML DETECTION MODELS - COMPUTER VISION MODULES
# ============================================================================

# ----------------------------------------------------------------------------
# 2.1 FACE DETECTION MODULE (Haar Cascade Classifier)
# ----------------------------------------------------------------------------
# Technology: Viola-Jones algorithm with Haar Cascade features
# Purpose: Detect human faces in video frames for security validation
# Algorithm: Cascade of boosted classifiers using Haar-like features
# Input: Grayscale image frame
# Output: List of face rectangles [(x, y, width, height), ...]
# Parameters:
#   - scaleFactor: 1.15 (image pyramid scale reduction)
#   - minNeighbors: 5 (minimum neighbors for detection confidence)
#   - minSize: (40, 40) pixels (minimum face size to detect)
# Model File: haarcascade_frontalface_default.xml (OpenCV pre-trained)
# Performance: ~30ms per frame on 640x480 image
from detection.face_detector import FaceDetector


# ----------------------------------------------------------------------------
# 2.2 FACE MASK DETECTION MODULE (PyTorch CNN Model)
# ----------------------------------------------------------------------------
# Model Type: Convolutional Neural Network (CNN) - Custom Architecture
# Framework: PyTorch
# Architecture Details:
#   - Conv Block 1: 3→32 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Conv Block 2: 32→64 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Conv Block 3: 64→128 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Flatten Layer
#   - FC Layer 1: 128 neurons, ReLU, Dropout(0.5)
#   - FC Layer 2: 2 neurons (output classes)
# 
# Classes (Binary Classification):
#   - Class 0: with_mask (Person wearing face mask/covering)
#   - Class 1: without_mask (Person with clear visible face)
# 
# Training Dataset: 7,553 images total
#   - Training Set: 6,042 images (80%)
#   - Validation Set: 1,511 images (20%)
#   - Image Size: 128x128 RGB
#   - Data Augmentation: Random rotation, flip, brightness adjustment
# 
# Model Performance:
#   - Validation Accuracy: 97.55%
#   - Training Accuracy: 99.12%
#   - Inference Time: ~15ms per face
# 
# Input: 128x128 RGB face image (extracted from detected face region)
# Output: (has_mask: bool, confidence: float)
# Confidence Threshold: 0.5 (configurable in config.json)
# Model File: models/mask_detector.pth
# 
# Usage in Security System:
#   - Detects if person is wearing face mask
#   - Triggers "mask_detected" violation if mask found
#   - Denies ATM access when mask detected
from detection.mask_detector_pytorch import MaskDetectorPyTorch as MaskDetector


# ----------------------------------------------------------------------------
# 2.3 HELMET DETECTION MODULE (PyTorch CNN Model)
# ----------------------------------------------------------------------------
# Model Type: Convolutional Neural Network (CNN) - Custom Architecture
# Framework: PyTorch
# Architecture Details:
#   - Conv Block 1: 3→32 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Conv Block 2: 32→64 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Conv Block 3: 64→128 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Flatten Layer
#   - FC Layer 1: 128 neurons, ReLU, Dropout(0.5)
#   - FC Layer 2: 2 neurons (output classes)
# 
# Classes (Binary Classification):
#   - Class 0: with_helmet (Person wearing helmet/head covering/cap)
#   - Class 1: without_helmet (Person with uncovered head)
# 
# Training Dataset: 3,925 images total
#   - Training Set: 3,140 images (80%)
#   - Validation Set: 785 images (20%)
#   - Image Size: 128x128 RGB
#   - Data Augmentation: Random rotation, flip, brightness adjustment
# 
# Model Performance:
#   - Validation Accuracy: 99.24%
#   - Training Accuracy: 99.87%
#   - Inference Time: ~15ms per face
# 
# Input: 128x128 RGB face image (extracted from detected face region)
# Output: (has_helmet: bool, confidence: float)
# Confidence Threshold: 0.4 (configurable in config.json)
# Model File: models/helmet_detector.pth
# 
# Usage in Security System:
#   - Detects if person is wearing helmet or head covering
#   - Triggers "helmet_detected" violation if helmet found
#   - Denies ATM access when helmet detected
#   - Can detect combined violations (mask + helmet)
from detection.helmet_detector_pytorch import HelmetDetectorPyTorch as HelmetDetector


# ----------------------------------------------------------------------------
# 2.4 PERSON COUNT DETECTION MODULE (PyTorch CNN Model)
# ----------------------------------------------------------------------------
# Model Type: Convolutional Neural Network (CNN) - Custom Architecture
# Framework: PyTorch
# Architecture Details:
#   - Conv Block 1: 3→32 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Conv Block 2: 32→64 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Conv Block 3: 64→128 filters, 3x3 kernel, ReLU, MaxPool(2x2)
#   - Flatten Layer
#   - FC Layer 1: 128 neurons, ReLU, Dropout(0.5)
#   - FC Layer 2: 2 neurons (output classes)
# 
# Classes (Binary Classification):
#   - Class 0: multiple_person (2 or more persons detected in frame)
#   - Class 1: single_person (Exactly 1 person in frame)
# 
# Training Dataset: 40 images total (Small dataset - can be improved)
#   - Training Set: 32 images (80%)
#   - Validation Set: 8 images (20%)
#   - Image Size: 128x128 RGB (full frame, not just face)
#   - Data Augmentation: Random rotation, flip, brightness adjustment
# 
# Model Performance:
#   - Validation Accuracy: 87.50%
#   - Training Accuracy: 93.75%
#   - Inference Time: ~15ms per frame
#   - Note: Accuracy can be improved with larger dataset
# 
# Input: 128x128 RGB full frame image (entire camera view)
# Output: (is_multiple: bool, confidence: float)
# Confidence Threshold: 0.3 (configurable in config.json)
# Model File: models/person_detector.pth
# 
# Usage in Security System:
#   - Detects if multiple people are present at ATM
#   - Works in conjunction with face count detection
#   - Triggers "multiple_people" violation if >1 person detected
#   - Denies ATM access when multiple people present
#   - Prevents unauthorized access attempts
# 
# Improvement Notes:
#   - Can be retrained with larger COCO person dataset
#   - See DATASET_DOWNLOAD_GUIDE.md for dataset expansion
#   - Current model trained on limited custom dataset
from detection.person_detector_pytorch import PersonDetectorPyTorch as PersonDetector



# ============================================================================
# SECTION 3: SECURITY & ACCESS CONTROL MODULES
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1 SECURITY STATUS MANAGER
# ----------------------------------------------------------------------------
# Purpose: Centralized security decision-making and access control
# 
# Core Responsibilities:
#   - Evaluate all detection results (face, mask, helmet, person count)
#   - Make final access decision (GRANT or DENY)
#   - Track violation types and security state
#   - Provide real-time security status to ATM and UI
# 
# Security Decision Logic:
#   
#   DENY ACCESS CONDITIONS:
#   1. face_count == 0 → "no_face_detected"
#      - No person visible in frame
#      - Camera may be blocked or no one present
#   
#   2. face_count >= 2 → "multiple_people"
#      - Multiple faces detected by Haar Cascade
#      - Potential unauthorized access attempt
#   
#   3. is_multiple_persons AND confidence > 0.3 → "multiple_people"
#      - CNN model confirms multiple people
#      - Secondary validation of multiple person scenario
#   
#   4. has_mask == True → "mask_detected"
#      - Face mask covering detected
#      - Identity cannot be verified
#   
#   5. has_helmet == True → "helmet_detected"
#      - Helmet or head covering detected
#      - Identity obscured
#   
#   6. has_mask AND has_helmet → "mask_and_helmet_detected"
#      - Both coverings detected simultaneously
#      - Maximum identity obscuration
#   
#   GRANT ACCESS CONDITIONS:
#   - face_count == 1 (exactly one face)
#   - has_mask == False (no mask detected)
#   - has_helmet == False (no helmet detected)
#   - is_multiple_persons == False (single person confirmed)
#   - All conditions must be satisfied simultaneously
# 
# Violation Types (String identifiers for audio alerts):
#   - "no_face_detected": No person visible
#   - "multiple_people": 2+ persons or faces detected
#   - "mask_detected": Face mask covering found
#   - "helmet_detected": Helmet/head covering found
#   - "mask_and_helmet_detected": Both coverings present
#   - None: No violation, access granted
# 
# Status Output Format:
#   {
#       'access_granted': bool,
#       'violation_type': str or None,
#       'face_count': int,
#       'has_mask': bool,
#       'has_helmet': bool,
#       'mask_confidence': float,
#       'helmet_confidence': float,
#       'is_multiple_persons': bool,
#       'person_confidence': float
#   }
from security.security_status_manager import SecurityStatusManager


# ----------------------------------------------------------------------------
# 3.2 AUDIO ALERT SYSTEM
# ----------------------------------------------------------------------------
# Purpose: Play voice warnings for security violations and system messages
# Technology: pygame.mixer for MP3 audio playback
# Architecture: Background thread with queue-based processing
# 
# Audio File Requirements:
#   Format: MP3
#   Location: audio/ directory (configurable)
#   Naming Convention: violation_type.mp3
# 
# Security Violation Audio Files (REQUIRED):
#   1. multiple_people.mp3
#      - Message: "Multiple people detected. Please ensure only one person..."
#      - Triggered: When 2+ persons detected
#      - Rate Limited: 8 seconds between plays
#   
#   2. mask_detected.mp3
#      - Message: "Face mask detected. Please remove mask for verification..."
#      - Triggered: When face mask detected
#      - Rate Limited: 8 seconds between plays
#   
#   3. helmet_detected.mp3
#      - Message: "Helmet detected. Please remove helmet for verification..."
#      - Triggered: When helmet/head covering detected
#      - Rate Limited: 8 seconds between plays
#   
#   4. mask_and_helmet.mp3
#      - Message: "Mask and helmet detected. Please remove all coverings..."
#      - Triggered: When both mask and helmet detected
#      - Rate Limited: 8 seconds between plays
# 
# System Message Audio Files (OPTIONAL):
#   5. welcome.mp3
#      - Message: "Welcome to secure ATM transactions"
#      - Triggered: When card inserted successfully
#      - NOT rate limited (plays immediately)
#   
#   6. transaction_complete.mp3
#      - Message: "Thank you, your transaction is successfully completed"
#      - Triggered: When withdrawal completed
#      - NOT rate limited (plays immediately)
# 
# Features:
#   - Background Thread Processing: Non-blocking audio playback
#   - Queue-Based System: FIFO queue for alert management
#   - Rate Limiting: 8 seconds between violation alerts (prevents spam)
#   - No Rate Limit for System Messages: Welcome/completion play immediately
#   - Automatic Error Recovery: Reinitializes on pygame errors
#   - Graceful Shutdown: Clean thread termination
# 
# Configuration (config.json):
#   - enabled: true/false (enable/disable audio system)
#   - rate_limit_seconds: 8 (seconds between violation alerts)
#   - audio_directory: "audio" (path to audio files)
# 
# Error Handling:
#   - Missing audio files: Logs warning, continues operation
#   - pygame initialization failure: Logs error, disables audio
#   - Playback errors: Attempts reinitialization
from audio.audio_alert_system import AudioAlertSystem


# ----------------------------------------------------------------------------
# 3.3 ATM STATE MANAGER
# ----------------------------------------------------------------------------
# Purpose: Manage ATM transaction flow, screens, and account state
# 
# ATM Screen States (State Machine):
#   1. welcome
#      - Initial screen when system starts
#      - Displays: "Insert Card to Begin"
#      - Action: Wait for card insertion
#      - Next: pin_entry (on card insert)
#   
#   2. pin_entry
#      - PIN input screen
#      - Displays: "Enter Your 4-Digit PIN"
#      - Validation: Must be exactly 4 digits
#      - Security: Checked before proceeding
#      - Next: menu (on correct PIN) or welcome (on failure)
#   
#   3. menu
#      - Main menu screen
#      - Options: Withdrawal, Balance Inquiry, Statement, Exit
#      - Security: Continuously monitored
#      - Next: withdrawal, balance, statement, or welcome
#   
#   4. withdrawal
#      - Amount entry screen
#      - Displays: "Enter Withdrawal Amount"
#      - Validation: Amount > 0, Amount <= balance, Amount <= max_withdrawal
#      - Security: Checked before processing
#      - Next: processing (on valid amount) or menu (on cancel)
#   
#   5. processing
#      - Transaction processing screen
#      - Displays: "Processing Transaction..."
#      - Duration: 2 seconds simulation
#      - Actions: Deduct amount, update balance, log transaction
#      - Next: complete (on success)
#   
#   6. complete
#      - Transaction confirmation screen
#      - Displays: "Transaction Complete", new balance
#      - Audio: Plays "transaction_complete.mp3"
#      - Duration: 3 seconds display
#      - Next: welcome (auto-reset)
# 
# Account Management:
#   - Initial Balance: $5,000 (configurable in config.json)
#   - Balance Tracking: Real-time balance updates
#   - Transaction History: Logged with timestamp
#   - Overdraft Protection: Prevents withdrawal > balance
# 
# Business Rules:
#   - Default PIN: "1234" (configurable in config.json)
#   - Maximum Withdrawal: $1,000 per transaction (configurable)
#   - Minimum Withdrawal: $1
#   - PIN Format: Exactly 4 digits
#   - Security Validation: Required at every transaction step
# 
# State Transitions:
#   welcome → pin_entry → menu → withdrawal → processing → complete → welcome
#   Any security violation → immediate return to welcome
from atm.atm_state_manager import ATMStateManager


# ----------------------------------------------------------------------------
# 3.4 TRANSACTION CONTROLLER
# ----------------------------------------------------------------------------
# Purpose: Coordinate security validation with ATM operations
# 
# Core Responsibilities:
#   - Validate security status before EVERY transaction operation
#   - Cancel transactions immediately on security violations
#   - Coordinate between ATM State Manager and Security Manager
#   - Ensure fail-safe operation (errors → deny access)
#   - Provide unified transaction API for Flask routes
# 
# Transaction Flow with Security Integration:
#   
#   1. Card Insertion Flow:
#      a. User clicks "Insert Card" button
#      b. Controller checks security status
#      c. If access_granted: Transition to pin_entry, play welcome audio
#      d. If access_denied: Return error with violation type
#   
#   2. PIN Entry Flow:
#      a. User enters 4-digit PIN
#      b. Controller checks security status
#      c. If access_denied: Cancel, return to welcome
#      d. If access_granted: Validate PIN
#      e. If PIN correct: Transition to menu
#      f. If PIN incorrect: Return error, stay on pin_entry
#   
#   3. Withdrawal Flow:
#      a. User enters withdrawal amount
#      b. Controller checks security status
#      c. If access_denied: Cancel transaction, return to welcome
#      d. If access_granted: Validate amount (>0, <=balance, <=max)
#      e. If valid: Process withdrawal, deduct balance
#      f. If invalid: Return error, stay on withdrawal
#      g. On success: Transition to complete, play completion audio
# 
# Security Validation Points:
#   - Before card insertion acceptance
#   - Before PIN validation
#   - Before withdrawal processing
#   - Continuous monitoring during all screens
# 
# Error Handling:
#   - Security violation: Immediate cancellation, return to welcome
#   - Invalid PIN: Error message, allow retry
#   - Insufficient balance: Error message, return to menu
#   - Amount validation failure: Error message, stay on withdrawal
#   - System errors: Fail-safe to deny access
# 
# API Response Format:
#   {
#       'success': bool,
#       'message': str,
#       'screen': str (current screen after operation),
#       'balance': float (current balance),
#       'violation_type': str or None (if security violation)
#   }
from atm.transaction_controller import TransactionController



# ============================================================================
# SECTION 4: VIDEO PROCESSING & STREAMING MODULE
# ============================================================================

# ----------------------------------------------------------------------------
# 4.1 VIDEO STREAM HANDLER
# ----------------------------------------------------------------------------
# Purpose: Generate MJPEG video stream with real-time security overlays
# Technology: OpenCV for image processing, Flask for HTTP streaming
# Stream Format: MJPEG (Motion JPEG) over HTTP
# 
# Video Processing Pipeline (Frame-by-Frame):
#   
#   1. Frame Capture (Input)
#      - Source: USB/Built-in camera (cv2.VideoCapture)
#      - Resolution: 640x480 pixels (configurable)
#      - Frame Rate: 30 FPS capture, 15 FPS processing (configurable)
#      - Color Space: BGR (OpenCV default)
#   
#   2. Face Detection (Haar Cascade)
#      - Convert frame to grayscale
#      - Apply Haar Cascade classifier
#      - Detect all faces in frame
#      - Output: List of face rectangles [(x,y,w,h), ...]
#      - Processing Time: ~30ms per frame
#   
#   3. ROI Extraction (Region of Interest)
#      - Extract each detected face region
#      - Resize to 128x128 pixels for CNN input
#      - Convert BGR to RGB color space
#      - Normalize pixel values (0-255 → 0-1)
#   
#   4. Image Preprocessing (CLAHE Enhancement)
#      - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
#      - Improves detection in poor lighting conditions
#      - Enhances facial features for better CNN accuracy
#      - Applied to grayscale version of face ROI
#   
#   5. Mask Detection (PyTorch CNN)
#      - Input: 128x128 RGB face image
#      - Forward pass through mask detection CNN
#      - Output: (has_mask: bool, confidence: float)
#      - Processing Time: ~15ms per face
#      - Threshold: 0.5 (configurable)
#   
#   6. Helmet Detection (PyTorch CNN)
#      - Input: 128x128 RGB face image
#      - Forward pass through helmet detection CNN
#      - Output: (has_helmet: bool, confidence: float)
#      - Processing Time: ~15ms per face
#      - Threshold: 0.4 (configurable)
#   
#   7. Person Count Detection (PyTorch CNN)
#      - Input: 128x128 RGB full frame (not just face)
#      - Forward pass through person detection CNN
#      - Output: (is_multiple: bool, confidence: float)
#      - Processing Time: ~15ms per frame
#      - Threshold: 0.3 (configurable)
#   
#   8. Security Status Evaluation
#      - Aggregate all detection results
#      - Update SecurityStatusManager
#      - Determine access_granted or access_denied
#      - Identify violation_type if denied
#   
#   9. Visual Overlays (Drawing on Frame)
#      - Face Bounding Boxes:
#        * Green box: Access granted (clear face)
#        * Red box: Access denied (violation detected)
#        * Thickness: 2 pixels
#      
#      - Status Text (Top-left corner):
#        * "ACCESS GRANTED" in green (if allowed)
#        * "ACCESS DENIED" in red (if violation)
#        * Font: cv2.FONT_HERSHEY_SIMPLEX
#        * Size: 0.7 scale
#      
#      - Face Count Display:
#        * "Faces: X" (number of detected faces)
#        * Color: White text
#      
#      - Violation Type Display:
#        * Shows specific violation (e.g., "MASK DETECTED")
#        * Color: Red text
#        * Position: Below status text
#      
#      - Confidence Scores:
#        * Mask confidence: "Mask: XX%"
#        * Helmet confidence: "Helmet: XX%"
#        * Color: Yellow text
#        * Position: Bottom-left corner
#   
#   10. Audio Alert Queuing
#       - Check for security violations
#       - Queue appropriate audio alert
#       - Rate limiting applied (8 seconds)
#       - Background thread handles playback
#   
#   11. MJPEG Encoding (Output)
#       - Encode frame as JPEG image
#       - Quality: 90% (configurable)
#       - Add multipart/x-mixed-replace boundary
#       - Stream to Flask HTTP response
#       - Yield to client browser
#   
#   12. Memory Cleanup
#       - Release frame memory
#       - Clear temporary buffers
#       - Prevent memory leaks
# 
# Performance Optimization:
#   - Frame Skip: Process every Nth frame for efficiency
#   - ROI Processing: Only analyze face regions, not full frame
#   - Batch Processing: Can process multiple faces in parallel
#   - Memory Management: Explicit cleanup after each frame
# 
# Error Handling:
#   - Camera disconnection: Attempt reconnection, show error frame
#   - Processing errors: Log error, continue with next frame
#   - Model inference errors: Fail-safe to deny access
#   - Stream interruption: Graceful recovery
# 
# Output Stream:
#   - Protocol: HTTP
#   - MIME Type: multipart/x-mixed-replace; boundary=frame
#   - Endpoint: /video
#   - Client: HTML <img> tag or video player
#   - Latency: <100ms end-to-end
from web.video_stream_handler import VideoStreamHandler



# ============================================================================
# SECTION 5: LOGGING SYSTEM CONFIGURATION
# ============================================================================

def setup_logging(log_level=logging.INFO):
    """
    Set up comprehensive logging system with file rotation.
    
    Logging Architecture:
    - Console Handler: INFO and above → stdout (for monitoring)
    - File Handler: ALL levels → logs/atm_system.log (for debugging)
    - Error Handler: ERROR and above → logs/atm_errors.log (for critical issues)
    
    Log Rotation:
    - Maximum File Size: 10 MB per log file
    - Backup Count: 5 files (keeps last 5 rotations)
    - Total Storage: ~50 MB for system logs, ~50 MB for error logs
    
    Log Format:
    - Console: [timestamp] [level] [module] message
    - File: [timestamp] [level] [module] [function:line] message
    
    Log Levels:
    - DEBUG: Detailed diagnostic information
    - INFO: General system information and state changes
    - WARNING: Warning messages for non-critical issues
    - ERROR: Error messages for failures
    - CRITICAL: Critical system failures
    
    Args:
        log_level: Minimum logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler for INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all levels with rotation
    file_handler = RotatingFileHandler(
        'logs/atm_system.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler for ERROR and above
    error_handler = RotatingFileHandler(
        'logs/atm_errors.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    logging.info("Logging system initialized")

logger = logging.getLogger(__name__)



# ============================================================================
# SECTION 6: CAMERA INITIALIZATION & MANAGEMENT
# ============================================================================

def initialize_camera(camera_config, max_retries=3):
    """
    Initialize camera with error handling and retry logic.
    
    Camera Configuration:
    - Device Index: 0 (default built-in camera) or 1, 2... (external cameras)
    - Resolution: 640x480 pixels (VGA)
    - Frame Rate: 30 FPS
    - Backend: Auto-detected by OpenCV (DirectShow on Windows, V4L2 on Linux)
    
    Initialization Steps:
    1. Open camera device using cv2.VideoCapture
    2. Verify camera opened successfully
    3. Set resolution (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT)
    4. Set frame rate (CAP_PROP_FPS)
    5. Validate settings applied correctly
    
    Retry Logic:
    - Maximum Attempts: 3 (configurable)
    - Retry Delay: 2 seconds between attempts
    - Failure Handling: Raise RuntimeError after max retries
    
    Common Issues:
    - Camera in use by another application
    - Insufficient permissions
    - Driver issues
    - Hardware disconnection
    
    Args:
        camera_config: Dictionary with camera configuration
            {
                'index': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            }
        max_retries: Maximum number of initialization attempts
    
    Returns:
        cv2.VideoCapture: Initialized camera object
    
    Raises:
        RuntimeError: If camera cannot be initialized after max_retries
    """
    camera_index = camera_config['index']
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to initialize camera (attempt {attempt + 1}/{max_retries})")
            camera = cv2.VideoCapture(camera_index)
            
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
                camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
                logger.info(f"Camera initialized successfully on attempt {attempt + 1}")
                return camera
            else:
                logger.warning(f"Camera failed to open on attempt {attempt + 1}")
                camera.release()
                
        except Exception as e:
            logger.error(f"Camera initialization error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)  # Wait before retry
    
    error_msg = f"Failed to initialize camera after {max_retries} attempts"
    logger.error(error_msg)
    raise RuntimeError(error_msg)



def reconnect_camera(camera, camera_config):
    """
    Attempt to reconnect to camera device after disconnection.
    
    Reconnection Scenarios:
    - Camera unplugged and replugged
    - Driver reset or recovery
    - System resume from sleep
    - Camera application conflict resolved
    
    Reconnection Process:
    1. Release existing camera object
    2. Wait briefly for device to stabilize
    3. Create new VideoCapture instance
    4. Verify device opened successfully
    5. Reapply configuration settings
    6. Return success status and new camera object
    
    Args:
        camera: Current camera object (may be disconnected or None)
        camera_config: Dictionary with camera configuration
    
    Returns:
        tuple: (success: bool, camera: cv2.VideoCapture or None)
            - (True, camera_object): Reconnection successful
            - (False, None): Reconnection failed
    """
    try:
        logger.info("Attempting camera reconnection...")
        
        # Release old camera
        if camera is not None:
            camera.release()
        
        # Try to reinitialize
        new_camera = cv2.VideoCapture(camera_config['index'])
        
        if new_camera.isOpened():
            new_camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
            new_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
            new_camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
            logger.info("Camera reconnected successfully")
            return True, new_camera
        else:
            logger.warning("Camera reconnection failed: device not opened")
            new_camera.release()
            return False, None
            
    except Exception as e:
        logger.error(f"Camera reconnection error: {e}")
        return False, None



# ============================================================================
# SECTION 7: CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_path='config.json'):
    """
    Load system configuration from JSON file.
    
    Configuration Structure (config.json):
    {
        "flask_port": 5004,
        "camera": {
            "index": 0,
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "detection": {
            "mask_threshold": 0.5,
            "helmet_threshold": 0.4,
            "person_threshold": 0.3,
            "max_faces_allowed": 2,
            "face_cascade": "haarcascade_frontalface_default.xml"
        },
        "audio": {
            "enabled": true,
            "rate_limit_seconds": 8,
            "audio_directory": "audio"
        },
        "atm": {
            "initial_balance": 5000.00,
            "max_withdrawal": 1000.00,
            "correct_pin": "1234"
        }
    }
    
    Configuration Parameters:
    
    Flask Server:
    - flask_port: HTTP server port (default: 5004)
    
    Camera Settings:
    - index: Camera device index (0=built-in, 1+=external)
    - width: Frame width in pixels (640)
    - height: Frame height in pixels (480)
    - fps: Target frame rate (30)
    
    Detection Thresholds:
    - mask_threshold: Mask detection confidence threshold (0.5)
    - helmet_threshold: Helmet detection confidence threshold (0.4)
    - person_threshold: Person count confidence threshold (0.3)
    - max_faces_allowed: Maximum faces before violation (2)
    - face_cascade: Haar Cascade XML file name
    
    Audio System:
    - enabled: Enable/disable audio alerts (true/false)
    - rate_limit_seconds: Seconds between violation alerts (8)
    - audio_directory: Path to audio files ("audio")
    
    ATM Business Rules:
    - initial_balance: Starting account balance ($5000.00)
    - max_withdrawal: Maximum withdrawal per transaction ($1000.00)
    - correct_pin: Valid PIN for authentication ("1234")
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        SystemExit: If configuration file not found or invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)



# ============================================================================
# SECTION 8: FLASK APPLICATION INITIALIZATION
# ============================================================================

def create_app():
    """
    Create and configure the Flask application with all system modules.
    
    Application Initialization Sequence:
    
    1. Create Flask App Instance
       - Initialize Flask with template and static folders
       - Set up application configuration
    
    2. Load Configuration
       - Read config.json file
       - Store in app.config for global access
    
    3. Initialize Camera
       - Open camera device with retry logic
       - Configure resolution and frame rate
       - Store camera object in app.config
    
    4. Initialize Detection Modules
       - Face Detector (Haar Cascade)
       - Mask Detector (PyTorch CNN)
       - Helmet Detector (PyTorch CNN)
       - Person Detector (PyTorch CNN)
       - Load trained models from models/ directory
    
    5. Initialize Security Manager
       - Create SecurityStatusManager instance
       - Set up access control logic
    
    6. Initialize Audio System (Optional)
       - Create AudioAlertSystem if enabled
       - Start background audio thread
       - Load audio files from audio/ directory
    
    7. Initialize ATM State Manager
       - Create ATMStateManager with initial balance
       - Set up state machine for transaction flow
    
    8. Initialize Transaction Controller
       - Create TransactionController
       - Link ATM state with security manager
    
    9. Initialize Video Stream Handler
       - Create VideoStreamHandler
       - Set up MJPEG streaming pipeline
    
    10. Register Flask Routes
        - Web interface routes
        - API endpoints
        - Video streaming endpoint
    
    All initialized modules are stored in app.config for access by routes.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    config = load_config()
    app.config['ATM_CONFIG'] = config
    
    logger.info("Flask application created")
    
    # Initialize camera capture with error handling
    camera_config = config['camera']
    camera = initialize_camera(camera_config)
    
    logger.info(f"Camera initialized: index={camera_config['index']}")
    
    # Initialize detection modules
    detection_config = config['detection']
    face_detector = FaceDetector(detection_config['face_cascade'])
    mask_detector = MaskDetector(detection_config['mask_threshold'])
    helmet_detector = HelmetDetector(detection_config['helmet_threshold'])
    person_detector = PersonDetector(threshold=detection_config['person_threshold'])
    logger.info("Detection modules initialized")
    
    # Initialize security status manager
    security_manager = SecurityStatusManager()
    logger.info("Security status manager initialized")
    
    # Initialize audio alert system
    audio_config = config['audio']
    audio_system = None
    if audio_config['enabled']:
        try:
            audio_system = AudioAlertSystem(audio_config['audio_directory'])
            logger.info("Audio alert system initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize audio system: {e}")
    
    # Initialize ATM state manager
    atm_config = config['atm']
    atm_state = ATMStateManager(atm_config['initial_balance'])
    logger.info(f"ATM state manager initialized with balance: ${atm_config['initial_balance']:.2f}")
    
    # Initialize transaction controller
    transaction_controller = TransactionController(atm_state, security_manager)
    logger.info("Transaction controller initialized")
    
    # Initialize video stream handler
    video_handler = VideoStreamHandler()
    logger.info("Video stream handler initialized")
    
    # Store modules in app config for access in routes
    app.config['camera'] = camera
    app.config['face_detector'] = face_detector
    app.config['mask_detector'] = mask_detector
    app.config['helmet_detector'] = helmet_detector
    app.config['person_detector'] = person_detector
    app.config['security_manager'] = security_manager
    app.config['audio_system'] = audio_system
    app.config['atm_state'] = atm_state
    app.config['transaction_controller'] = transaction_controller
    app.config['video_handler'] = video_handler
    
    # Register routes
    register_routes(app)
    
    return app



# ============================================================================
# SECTION 9: FLASK ROUTES - WEB INTERFACE & API ENDPOINTS
# ============================================================================

def register_routes(app):
    """
    Register all Flask routes for web interface and API.
    
    Route Categories:
    1. Web Interface Routes (HTML pages)
    2. Video Streaming Routes (MJPEG stream)
    3. API Routes (JSON endpoints for ATM operations)
    """
    
    # ------------------------------------------------------------------------
    # 9.1 WEB INTERFACE ROUTE
    # ------------------------------------------------------------------------
    
    @app.route('/')
    def index():
        """
        Serve main ATM interface HTML page.
        
        Endpoint: GET /
        Response: HTML page (templates/index.html)
        
        Page Components:
        - Video feed display (shows camera with overlays)
        - ATM screen interface (transaction UI)
        - Security status indicators
        - Transaction controls (buttons, input fields)
        
        The HTML page uses JavaScript to:
        - Display video stream from /video endpoint
        - Poll /api/status for ATM and security updates
        - Send transaction requests to API endpoints
        - Update UI based on current screen state
        """
        return render_template('index.html')
    
    # ------------------------------------------------------------------------
    # 9.2 VIDEO STREAMING ROUTE
    # ------------------------------------------------------------------------
    
    @app.route('/video')
    def video():
        """
        Stream video feed with security overlays (MJPEG format).
        
        Endpoint: GET /video
        Response: multipart/x-mixed-replace MJPEG stream
        
        Stream Processing:
        1. Capture frame from camera
        2. Detect faces using Haar Cascade
        3. Run mask detection on each face
        4. Run helmet detection on each face
        5. Run person count detection on full frame
        6. Update security status
        7. Draw visual overlays (boxes, text, status)
        8. Queue audio alerts for violations
        9. Encode frame as JPEG
        10. Yield frame to HTTP stream
        11. Repeat continuously
        
        Error Handling:
        - Camera disconnection: Attempt reconnection, show error frame
        - Processing errors: Log error, continue with next frame
        - Stream interruption: Graceful recovery
        
        Performance:
        - Target: 15 FPS processing
        - Latency: <100ms end-to-end
        - Bandwidth: ~500 KB/s at 640x480
        """
        camera = app.config['camera']
        face_detector = app.config['face_detector']
        mask_detector = app.config['mask_detector']
        helmet_detector = app.config['helmet_detector']
        person_detector = app.config['person_detector']
        security_manager = app.config['security_manager']
        video_handler = app.config['video_handler']
        audio_system = app.config['audio_system']
        camera_config = app.config['ATM_CONFIG']['camera']
        
        def generate():
            """
            Generate frames and queue audio alerts with camera error handling.
            
            This generator function runs continuously, yielding video frames
            to the HTTP response. It handles camera errors and attempts
            reconnection if the camera becomes unavailable.
            """
            nonlocal camera
            last_reconnect_attempt = 0
            reconnect_interval = 5  # seconds
            
            while True:
                try:
                    # Check if camera is available
                    if camera is None or not camera.isOpened():
                        logger.error("Camera not available, setting security to denied")
                        security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0, 'person': 0.0}, False)
                        
                        # Attempt reconnection if enough time has passed
                        current_time = time.time()
                        if current_time - last_reconnect_attempt >= reconnect_interval:
                            last_reconnect_attempt = current_time
                            success, new_camera = reconnect_camera(camera, camera_config)
                            if success:
                                camera = new_camera
                                app.config['camera'] = camera
                                logger.info("Camera reconnected, resuming video stream")
                                continue
                        
                        # Yield error frame
                        yield video_handler.generate_error_frame("Camera Error: Reconnecting...")
                        time.sleep(1)
                        continue
                    
                    # Generate frames normally
                    for frame_data in video_handler.generate_frames(
                        camera, face_detector, mask_detector, helmet_detector, security_manager, person_detector
                    ):
                        # Queue audio alert if there's a violation and audio is enabled
                        if audio_system:
                            security_status = security_manager.get_status()
                            if not security_status['access_granted'] and security_status['violation_type']:
                                audio_system.queue_alert(security_status['violation_type'])
                        
                        yield frame_data
                        
                except Exception as e:
                    logger.error(f"Video stream error: {e}")
                    security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
                    yield video_handler.generate_error_frame(f"Stream Error: {str(e)}")
                    time.sleep(1)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    
    # ------------------------------------------------------------------------
    # 9.3 API ENDPOINTS - ATM OPERATIONS
    # ------------------------------------------------------------------------
    
    @app.route('/api/status')
    def api_status():
        """
        Get current ATM and security status.
        
        Endpoint: GET /api/status
        Response: JSON
        
        Response Format:
        {
            "screen": "welcome|pin_entry|menu|withdrawal|processing|complete",
            "balance": 5000.00,
            "security": {
                "access_granted": true|false,
                "violation_type": "mask_detected|helmet_detected|multiple_people|...|null",
                "face_count": 1,
                "has_mask": false,
                "has_helmet": false,
                "mask_confidence": 0.95,
                "helmet_confidence": 0.12,
                "is_multiple_persons": false,
                "person_confidence": 0.15
            }
        }
        
        Usage:
        - Polled by frontend JavaScript every 500ms
        - Updates UI based on current screen
        - Shows security status indicators
        - Displays violation warnings
        """
        atm_state = app.config['atm_state']
        security_manager = app.config['security_manager']
        
        # Get ATM state
        state = atm_state.get_state()
        
        # Get security status
        security_status = security_manager.get_status()
        
        # Combine into response
        response = {
            'screen': state['current_screen'],
            'balance': state['balance'],
            'security': {
                'access_granted': security_status['access_granted'],
                'violation_type': security_status['violation_type'],
                'face_count': security_status['face_count'],
                'has_mask': security_status['has_mask'],
                'has_helmet': security_status['has_helmet'],
                'mask_confidence': security_status['mask_confidence'],
                'helmet_confidence': security_status['helmet_confidence']
            }
        }
        
        return jsonify(response)
    
    @app.route('/api/insert_card', methods=['POST'])
    def api_insert_card():
        """
        Handle card insertion (start transaction).
        
        Endpoint: POST /api/insert_card
        Request: Empty JSON {}
        Response: JSON
        
        Process:
        1. Check security status
        2. If access granted:
           - Transition to pin_entry screen
           - Play welcome audio
           - Return success
        3. If access denied:
           - Return error with violation type
           - Stay on welcome screen
        
        Response Format (Success):
        {
            "success": true,
            "message": "Card accepted. Please enter PIN.",
            "screen": "pin_entry",
            "balance": 5000.00
        }
        
        Response Format (Failure):
        {
            "success": false,
            "message": "Access denied: Multiple people detected",
            "screen": "welcome",
            "balance": 5000.00,
            "violation_type": "multiple_people"
        }
        """
        transaction_controller = app.config['transaction_controller']
        
        result = transaction_controller.handle_card_insertion()
        
        # Play welcome audio if successful and audio enabled
        if result['success'] and app.config['audio_system']:
            app.config['audio_system'].play_system_message('welcome')
        
        return jsonify(result)
    
    @app.route('/api/enter_pin', methods=['POST'])
    def api_enter_pin():
        """
        Handle PIN entry and validation.
        
        Endpoint: POST /api/enter_pin
        Request: {"pin": "1234"}
        Response: JSON
        
        Process:
        1. Check security status
        2. If access denied:
           - Cancel transaction
           - Return to welcome screen
        3. If access granted:
           - Validate PIN format (4 digits)
           - Check PIN correctness
           - If correct: Transition to menu
           - If incorrect: Return error, stay on pin_entry
        
        Response Format (Success):
        {
            "success": true,
            "message": "PIN accepted",
            "screen": "menu",
            "balance": 5000.00
        }
        
        Response Format (Failure - Wrong PIN):
        {
            "success": false,
            "message": "Incorrect PIN",
            "screen": "pin_entry",
            "balance": 5000.00
        }
        
        Response Format (Failure - Security Violation):
        {
            "success": false,
            "message": "Security violation: mask_detected",
            "screen": "welcome",
            "balance": 5000.00,
            "violation_type": "mask_detected"
        }
        """
        transaction_controller = app.config['transaction_controller']
        
        data = request.get_json()
        pin = data.get('pin', '')
        
        result = transaction_controller.handle_pin_entry(pin)
        
        return jsonify(result)
    
    @app.route('/api/select_withdrawal', methods=['POST'])
    def api_select_withdrawal():
        """
        Navigate to withdrawal screen from menu.
        
        Endpoint: POST /api/select_withdrawal
        Request: Empty JSON {}
        Response: JSON
        
        Process:
        1. Transition from menu to withdrawal screen
        2. Return new screen state
        
        Response Format:
        {
            "success": true,
            "message": "Enter withdrawal amount",
            "screen": "withdrawal",
            "balance": 5000.00
        }
        """
        atm_state = app.config['atm_state']
        
        result = atm_state.select_withdrawal()
        
        return jsonify(result)

    
    @app.route('/api/withdraw', methods=['POST'])
    def api_withdraw():
        """
        Process withdrawal transaction.
        
        Endpoint: POST /api/withdraw
        Request: {"amount": 100.00}
        Response: JSON
        
        Process:
        1. Check security status
        2. If access denied:
           - Cancel transaction
           - Return to welcome screen
        3. If access granted:
           - Validate amount (>0, <=balance, <=max_withdrawal)
           - If valid:
             * Deduct amount from balance
             * Transition to processing screen
             * Simulate processing (2 seconds)
             * Transition to complete screen
             * Play completion audio
             * Auto-reset to welcome after 3 seconds
           - If invalid:
             * Return error
             * Stay on withdrawal screen
        
        Response Format (Success):
        {
            "success": true,
            "message": "Withdrawal successful",
            "screen": "complete",
            "balance": 4900.00,
            "amount": 100.00
        }
        
        Response Format (Failure - Insufficient Balance):
        {
            "success": false,
            "message": "Insufficient balance",
            "screen": "withdrawal",
            "balance": 5000.00
        }
        
        Response Format (Failure - Security Violation):
        {
            "success": false,
            "message": "Security violation: helmet_detected",
            "screen": "welcome",
            "balance": 5000.00,
            "violation_type": "helmet_detected"
        }
        """
        transaction_controller = app.config['transaction_controller']
        
        data = request.get_json()
        amount = data.get('amount', 0)
        
        result = transaction_controller.handle_withdrawal(amount)
        
        # Play completion audio if successful and audio enabled
        if result['success'] and app.config['audio_system']:
            app.config['audio_system'].play_system_message('transaction_complete')
        
        return jsonify(result)
    
    @app.route('/api/reset', methods=['POST'])
    def api_reset():
        """
        Reset ATM to welcome screen (cancel transaction).
        
        Endpoint: POST /api/reset
        Request: Empty JSON {}
        Response: JSON
        
        Process:
        1. Reset ATM state to welcome screen
        2. Clear any pending transactions
        3. Return to initial state
        
        Response Format:
        {
            "success": true,
            "message": "ATM reset to welcome screen",
            "screen": "welcome",
            "balance": 5000.00
        }
        
        Usage:
        - Cancel button on any screen
        - Timeout after inactivity
        - Security violation detected
        - Transaction completion
        """
        atm_state = app.config['atm_state']
        
        result = atm_state.reset()
        
        return jsonify(result)



# ============================================================================
# SECTION 10: RESOURCE CLEANUP & SHUTDOWN
# ============================================================================

def cleanup_resources(app):
    """
    Gracefully cleanup all system resources on shutdown.
    
    Cleanup Sequence:
    1. Shutdown audio system (stop background thread)
    2. Release camera device
    3. Destroy OpenCV windows
    4. Log cleanup completion
    
    This function is called:
    - On normal application exit
    - On Ctrl+C (SIGINT)
    - On system termination (SIGTERM)
    - On fatal errors
    
    Ensures:
    - No resource leaks
    - Clean thread termination
    - Proper device release
    - Graceful shutdown
    
    Args:
        app: Flask application instance with config containing system modules
    """
    logger.info("Starting resource cleanup...")
    
    # Shutdown audio system first (stops background thread)
    audio_system = app.config.get('audio_system')
    if audio_system:
        try:
            audio_system.shutdown()
            logger.info("Audio system shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down audio system: {e}")
    
    # Release camera device
    camera = app.config.get('camera')
    if camera:
        try:
            if camera.isOpened():
                camera.release()
            logger.info("Camera device released")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")
    
    # Destroy all OpenCV windows
    try:
        cv2.destroyAllWindows()
        logger.info("OpenCV windows destroyed")
    except Exception as e:
        logger.error(f"Error destroying OpenCV windows: {e}")
    
    logger.info("Resource cleanup complete")


# ============================================================================
# SECTION 11: MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the ATM Security System.
    
    Startup Sequence:
    1. Initialize logging system
    2. Log startup banner
    3. Register signal handlers for graceful shutdown
    4. Create Flask application
    5. Initialize all system modules
    6. Start Flask HTTP server
    7. Run until shutdown signal received
    8. Cleanup resources
    9. Exit
    
    Signal Handling:
    - SIGINT (Ctrl+C): Graceful shutdown
    - SIGTERM (kill): Graceful shutdown
    
    Error Handling:
    - Initialization errors: Log and exit
    - Runtime errors: Log and attempt recovery
    - Fatal errors: Cleanup and exit
    
    Server Configuration:
    - Host: 0.0.0.0 (accessible from network)
    - Port: 5004 (configurable in config.json)
    - Debug: False (production mode)
    - Threaded: True (handle multiple requests)
    
    Access:
    - Local: http://localhost:5004
    - Network: http://<your-ip>:5004
    """
    # Set up logging first
    setup_logging(log_level=logging.INFO)
    
    logger.info("=" * 80)
    logger.info("ATM SECURITY SYSTEM - STARTING")
    logger.info("=" * 80)
    logger.info("Version: 3.0 (Complete Integrated)")
    logger.info("Date: November 19, 2025")
    logger.info("=" * 80)
    
    app = None
    
    def signal_handler(signum, frame):
        """
        Handle shutdown signals gracefully.
        
        Called when:
        - User presses Ctrl+C (SIGINT)
        - System sends termination signal (SIGTERM)
        
        Actions:
        - Log shutdown initiation
        - Cleanup all resources
        - Exit with code 0
        """
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if app:
            cleanup_resources(app)
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create Flask application
        logger.info("Creating Flask application...")
        app = create_app()
        
        # Get configuration
        config = app.config['ATM_CONFIG']
        port = config.get('flask_port', 5004)
        
        logger.info("=" * 80)
        logger.info(f"Starting Flask server on port {port}")
        logger.info(f"Access the ATM interface at: http://localhost:{port}")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 80)
        
        # Run Flask application
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal (Ctrl+C)")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        
    finally:
        # Always cleanup resources on exit
        if app:
            cleanup_resources(app)
        logger.info("=" * 80)
        logger.info("ATM Security System shutdown complete")
        logger.info("=" * 80)
        sys.exit(0)


# ============================================================================
# SECTION 12: SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Script execution entry point.
    
    When this file is run directly (not imported), execute the main() function.
    
    Usage:
    1. Command Line: python app_final_complete.py
    2. IDLE: Open file and press F5
    3. IDE: Run this file
    4. Batch File: START_ATM.bat (Windows)
    
    Requirements:
    - Python 3.7+
    - All dependencies installed (see requirements.txt)
    - Camera device available
    - Trained models in models/ directory
    - Audio files in audio/ directory (optional)
    - Configuration file: config.json
    """
    main()


# ============================================================================
# END OF FILE
# ============================================================================
