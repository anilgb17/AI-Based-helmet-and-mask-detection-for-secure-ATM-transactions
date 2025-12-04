"""
================================================================================
ATM SECURITY SYSTEM - COMPLETE APPLICATION WITH DETAILED COMMENTS
================================================================================

This is the main application file that integrates all components of the ATM
Security System including:
- Face Detection (Haar Cascade)
- Mask Detection (PyTorch CNN - with_mask vs without_mask)
- Helmet Detection (PyTorch CNN - with_helmet vs without_helmet)
- Person Detection (PyTorch CNN - single_person vs multiple_person)
- Security Violation Management
- ATM Transaction Processing
- Video Streaming with Overlays
- Audio Alert System

Author: ATM Security System Development Team
Version: 1.0
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

# ============================================================================
# SECTION 2: DETECTION MODULES - COMPUTER VISION & DEEP LEARNING
# ============================================================================

# Face Detection Module (Haar Cascade Classifier)
# Detects human faces in video frames for security monitoring
from detection.face_detector import FaceDetector

# Mask Detection Module (PyTorch CNN Model)
# Binary Classification: with_mask vs without_mask
# Trained on 7,553 images with 97.55% validation accuracy
from detection.mask_detector_pytorch import MaskDetectorPyTorch as MaskDetector

# Helmet Detection Module (PyTorch CNN Model)
# Binary Classification: with_helmet vs without_helmet
# Trained on 3,925 images with 99.24% validation accuracy
from detection.helmet_detector_pytorch import HelmetDetectorPyTorch as HelmetDetector

# Person Detection Module (PyTorch CNN Model)
# Binary Classification: single_person vs multiple_person
# Trained on 40 images with 87.50% validation accuracy
from detection.person_detector_pytorch import PersonDetectorPyTorch as PersonDetector


# ============================================================================
# SECTION 3: SECURITY & ATM MODULES
# ============================================================================

# Security Status Manager
# Manages access control decisions based on detection results
# Implements security logic: deny access for violations, grant for clear face
from security.security_status_manager import SecurityStatusManager

# Audio Alert System
# Plays voice warnings for security violations with rate limiting
# Supports 4 violation types: multiple_people, mask_detected, helmet_detected, mask_and_helmet
from audio.audio_alert_system import AudioAlertSystem

# ATM State Manager
# Manages ATM screens, transaction flow, and account balance
# Screens: welcome, pin_entry, menu, withdrawal, processing, complete
from atm.atm_state_manager import ATMStateManager

# Transaction Controller
# Coordinates security validation with ATM operations
# Ensures every transaction step validates security status
from atm.transaction_controller import TransactionController

# ============================================================================
# SECTION 4: VIDEO STREAMING MODULE
# ============================================================================

# Video Stream Handler
# Generates MJPEG video stream with security overlays
# Draws face boxes, status text, and violation indicators
from web.video_stream_handler import VideoStreamHandler

# ============================================================================
# SECTION 5: LOGGING CONFIGURATION
# ============================================================================

from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO):
    """
    Configure comprehensive logging system with file rotation.
    
    Creates three log outputs:
    1. Console: INFO and above for real-time monitoring
    2. File (atm_system.log): All levels with rotation (10MB, 5 backups)
    3. Error File (atm_errors.log): ERROR and above for troubleshooting
    
    Args:
        log_level: Minimum logging level (default: INFO)
    """
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    
    # Console handler - displays INFO and above to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - logs all levels with rotation
    file_handler = RotatingFileHandler(
        'logs/atm_system.log',
        maxBytes=10*1024*1024,  # 10MB per file
        backupCount=5           # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler - logs ERROR and above separately
    error_handler = RotatingFileHandler(
        'logs/atm_errors.log',
        maxBytes=10*1024*1024,  # 10MB per file
        backupCount=5           # Keep 5 backup files
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    logging.info("Logging system initialized successfully")

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 6: CAMERA INITIALIZATION & MANAGEMENT
# ============================================================================

def initialize_camera(camera_config, max_retries=3):
    """
    Initialize camera with error handling and retry logic.
    
    Attempts to open the camera device and configure resolution/FPS.
    Retries up to max_retries times with 2-second delays between attempts.
    
    Args:
        camera_config: Dictionary with 'index', 'width', 'height', 'fps'
        max_retries: Maximum number of initialization attempts (default: 3)
    
    Returns:
        cv2.VideoCapture: Initialized camera object
    
    Raises:
        RuntimeError: If camera cannot be initialized after max_retries
    
    Example camera_config:
        {
            "index": 0,        # Camera device index (0 = default)
            "width": 640,      # Frame width in pixels
            "height": 480,     # Frame height in pixels
            "fps": 30          # Target frames per second
        }
    """
    camera_index = camera_config['index']
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Camera initialization attempt {attempt + 1}/{max_retries}")
            camera = cv2.VideoCapture(camera_index)
            
            if camera.isOpened():
                # Configure camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
                camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
                logger.info(f"Camera initialized successfully: {camera_config['width']}x{camera_config['height']} @ {camera_config['fps']}fps")
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
    
    This function is called when camera read fails, attempting to
    re-establish connection without restarting the application.
    
    Args:
        camera: Current camera object (may be disconnected)
        camera_config: Dictionary with camera configuration
    
    Returns:
        tuple: (success: bool, camera: cv2.VideoCapture or None)
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
    
    Configuration includes:
    - Camera settings (index, resolution, FPS)
    - Detection thresholds (mask, helmet)
    - Audio settings (enabled, rate limiting)
    - ATM parameters (balance, PIN, withdrawal limits)
    - Flask server port
    
    Args:
        config_path: Path to JSON configuration file (default: 'config.json')
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        SystemExit: If config file not found or invalid JSON
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
# SECTION 8: FLASK APPLICATION CREATION & MODULE INITIALIZATION
# ============================================================================

def create_app():
    """
    Create and configure the Flask application with all modules.
    
    This function initializes all system components in the correct order:
    1. Load configuration
    2. Initialize camera
    3. Initialize detection modules (face, mask, helmet, person)
    4. Initialize security manager
    5. Initialize audio alert system
    6. Initialize ATM state manager
    7. Initialize transaction controller
    8. Initialize video stream handler
    9. Register Flask routes
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    config = load_config()
    app.config['ATM_CONFIG'] = config
    logger.info("Flask application created")
    
    # ========================================================================
    # STEP 2: Initialize Camera
    # ========================================================================
    camera_config = config['camera']
    camera = initialize_camera(camera_config)
    logger.info(f"Camera initialized: index={camera_config['index']}, "
                f"resolution={camera_config['width']}x{camera_config['height']}")
    
    # ========================================================================
    # STEP 3: Initialize Detection Modules
    # ========================================================================
    detection_config = config['detection']
    
    # Face Detector - Haar Cascade Classifier
    # Detects faces in video frames using Viola-Jones algorithm
    face_detector = FaceDetector(detection_config['face_cascade'])
    logger.info("Face detector initialized (Haar Cascade)")
    
    # Mask Detector - PyTorch CNN Model
    # Binary classification: with_mask (class 0) vs without_mask (class 1)
    # Model: 3 conv blocks + fully connected layers
    # Accuracy: 97.55% on validation set
    mask_detector = MaskDetector(detection_config['mask_threshold'])
    logger.info(f"Mask detector initialized (threshold={detection_config['mask_threshold']})")
    
    # Helmet Detector - PyTorch CNN Model
    # Binary classification: with_helmet (class 0) vs without_helmet (class 1)
    # Model: 3 conv blocks + fully connected layers
    # Accuracy: 99.24% on validation set
    helmet_detector = HelmetDetector(detection_config['helmet_threshold'])
    logger.info(f"Helmet detector initialized (threshold={detection_config['helmet_threshold']})")

    
    # Person Detector - PyTorch CNN Model
    # Binary classification: multiple_person (class 0) vs single_person (class 1)
    # Model: 3 conv blocks + fully connected layers
    # Accuracy: 87.50% on validation set
    # Used to detect coordinated fraud attempts with multiple people
    person_detector = PersonDetector(threshold=0.5)
    logger.info("Person detector initialized (threshold=0.5)")
    
    logger.info("All detection modules initialized successfully")
    
    # ========================================================================
    # STEP 4: Initialize Security Status Manager
    # ========================================================================
  
    security_manager = SecurityStatusManager()
    logger.info("Security status manager initialized")
    
    # ========================================================================
    # STEP 5: Initialize Audio Alert System
    # ========================================================================
  
    audio_config = config['audio']
    audio_system = None
    if audio_config['enabled']:
        try:
            audio_system = AudioAlertSystem(audio_config['audio_directory'])
            logger.info(f"Audio alert system initialized (rate_limit={audio_config['rate_limit_seconds']}s)")
        except Exception as e:
            logger.warning(f"Failed to initialize audio system: {e}")
            logger.warning("Continuing without audio alerts")

    
    # ========================================================================
    # STEP 6: Initialize ATM State Manager
    # ========================================================================
   
    atm_config = config['atm']
    atm_state = ATMStateManager(atm_config['initial_balance'])
    logger.info(f"ATM state manager initialized (balance=${atm_config['initial_balance']:.2f}, "
                f"max_withdrawal=${atm_config['max_withdrawal']:.2f})")
    
    # ========================================================================
    # STEP 7: Initialize Transaction Controller
    # ========================================================================
  
    transaction_controller = TransactionController(atm_state, security_manager)
    logger.info("Transaction controller initialized")
    
    # ========================================================================
    # STEP 8: Initialize Video Stream Handler
    # ========================================================================
   
    video_handler = VideoStreamHandler()
    logger.info("Video stream handler initialized")

    
    # ========================================================================
    # STEP 9: Store Modules in App Config
    # ========================================================================
    # Store all initialized modules in Flask app config for access in routes
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
    
    # ========================================================================
    # STEP 10: Register Flask Routes
    # ========================================================================
    register_routes(app)
    
    logger.info("Flask application fully configured and ready")
    return app

# ============================================================================
# SECTION 9: FLASK ROUTE DEFINITIONS
# ============================================================================

def register_routes(app):
    """
    Register all Flask routes for the ATM Security System.
    
    Routes:
    - GET  /              : Serve main ATM interface HTML
    - GET  /video         : Stream live video with security overlays (MJPEG)
    - GET  /api/status    : Get current ATM and security status (JSON)
    - POST /api/insert_card    : Handle card insertion
    - POST /api/enter_pin      : Handle PIN entry
    - POST /api/select_withdrawal : Navigate to withdrawal screen
    - POST /api/withdraw       : Process withdrawal transaction
    - POST /api/reset          : Reset ATM to welcome screen
    """
    
    # ========================================================================
    # ROUTE 1: Main ATM Interface
    # ========================================================================
    @app.route('/')
    def index():
        """
        Serve the main ATM interface HTML page.
        
        Returns:
            HTML: Rendered index.html template with ATM interface
        """
        return render_template('index.html')

    
    # ========================================================================
    # ROUTE 2: Video Streaming with Security Overlays
    # ========================================================================
    @app.route('/video')
    def video():
        """
        Stream live video feed with security overlays in MJPEG format.
        
        Video Processing Flow:
        1. Capture frame from camera (640x480 @ 30fps)
        2. Face Detection: Detect faces using Haar Cascade
        3. ROI Extraction: Extract face region for detailed analysis
        4. Preprocessing: Apply CLAHE enhancement for better detection
        5. Mask Detection: Run PyTorch CNN (with_mask vs without_mask)
        6. Helmet Detection: Run PyTorch CNN (with_helmet vs without_helmet)
        7. Person Detection: Run PyTorch CNN (single vs multiple persons)
        8. Security Evaluation: Update security status based on detections
        9. Visual Overlays: Draw face boxes, status text, violations
        10. Audio Alerts: Queue audio warnings for violations
        11. MJPEG Encoding: Encode frame as JPEG and stream
        
        Security Status Updates:
        - face_count: Number of faces detected (0, 1, or >1)
        - has_mask: Boolean indicating mask detection
        - has_helmet: Boolean indicating helmet detection
        - is_multiple_persons: Boolean from person detection CNN
        - mask_confidence: Float 0.0-1.0 (threshold: 0.5)
        - helmet_confidence: Float 0.0-1.0 (threshold: 0.4)
        - person_confidence: Float 0.0-1.0 (threshold: 0.5)
        
        Returns:
            Response: MJPEG video stream (multipart/x-mixed-replace)
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
            Generator function that yields video frames with security processing.
            
            Handles:
            - Camera disconnections with automatic reconnection
            - Detection errors with fail-safe defaults
            - Audio alert queueing for violations
            - Memory cleanup after each frame
            - Welcome message on first frame
            """
            nonlocal camera
            last_reconnect_attempt = 0
            reconnect_interval = 5  # seconds
            welcome_played = False  # Track if welcome message has been played
            
            while True:
                try:
                    # Check camera availability
                    if camera is None or not camera.isOpened():
                        logger.error("Camera not available, denying access")
                        security_manager.update_status(0, False, False, 
                                                      {'mask': 0.0, 'helmet': 0.0, 'person': 0.0}, False)
                        
                        # Attempt reconnection
                        current_time = time.time()
                        if current_time - last_reconnect_attempt >= reconnect_interval:
                            last_reconnect_attempt = current_time
                            success, new_camera = reconnect_camera(camera, camera_config)
                            if success:
                                camera = new_camera
                                app.config['camera'] = camera
                                logger.info("Camera reconnected, resuming video stream")
                                continue
                        
                        yield video_handler.generate_error_frame("Camera Error: Reconnecting...")
                        time.sleep(1)
                        continue
                    
                    # Play welcome message on first successful frame
                    if not welcome_played and audio_system:
                        audio_system.play_immediate('welcome')
                        welcome_played = True
                        logger.info("Welcome message played: 'Welcome to secure ATM transactions'")
                    
                    # Generate frames with detection and overlays
                    for frame_data in video_handler.generate_frames(
                        camera, face_detector, mask_detector, helmet_detector, 
                        security_manager, person_detector
                    ):
                        # Queue audio alert if violation detected and audio enabled
                        if audio_system:
                            security_status = security_manager.get_status()
                            if not security_status['access_granted'] and security_status['violation_type']:
                                # Queue appropriate audio alert based on violation type
                                audio_system.queue_alert(security_status['violation_type'])
                        
                        yield frame_data
                        
                except Exception as e:
                    logger.error(f"Video stream error: {e}")
                    security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0})
                    yield video_handler.generate_error_frame(f"Stream Error: {str(e)}")
                    time.sleep(1)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # ========================================================================
    # ROUTE 3: System Status API
    # ========================================================================
    @app.route('/api/status')
    def api_status():
        """
        Get current ATM and security status.
        
        Returns JSON with:
        - ATM state (screen, balance)
        - Security status (access_granted, violation_type)
        - Detection results (face_count, has_mask, has_helmet)
        - Confidence scores (mask_confidence, helmet_confidence)
        
        Returns:
            JSON: {
                "screen": "welcome|pin_entry|menu|withdrawal|processing|complete",
                "balance": 5000.00,
                "security": {
                    "access_granted": true|false,
                    "violation_type": null|"no_face_detected"|"multiple_people"|
                                     "mask_detected"|"helmet_detected"|
                                     "mask_and_helmet_detected",
                    "face_count": 0|1|2+,
                    "has_mask": true|false,
                    "has_helmet": true|false,
                    "mask_confidence": 0.0-1.0,
                    "helmet_confidence": 0.0-1.0
                }
            }
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

    
    # ========================================================================
    # ROUTE 4: Card Insertion API
    # ========================================================================
    @app.route('/api/insert_card', methods=['POST'])
    def api_insert_card():
        """
        Handle ATM card insertion request.
        
        Security Validation:
        - Checks current security status before allowing card insertion
        - Denies if no face detected
        - Denies if multiple people detected
        - Denies if mask detected
        - Denies if helmet detected
        - Grants only if single clear face detected
        
        Success Flow:
        1. Validate security status
        2. Transition from 'welcome' to 'pin_entry' screen
        3. Clear PIN input buffer
        4. Return success response
        
        Failure Flow:
        1. Security violation detected
        2. Remain on 'welcome' screen
        3. Return error with violation message
        
        Returns:
            JSON: {
                "success": true|false,
                "message": "Card inserted successfully" | "Access denied: <violation>",
                "screen": "welcome|pin_entry"
            }
        """
        transaction_controller = app.config['transaction_controller']
        
        result = transaction_controller.handle_card_insertion()
        
        return jsonify(result)
    
    # ========================================================================
    # ROUTE 5: PIN Entry API
    # ========================================================================
    @app.route('/api/enter_pin', methods=['POST'])
    def api_enter_pin():
        """
        Handle PIN entry and validation.
        
        Security Validation:
        - Checks security status before validating PIN
        - Cancels transaction if violation detected during PIN entry
        
        PIN Validation:
        - Expects 4-digit PIN string
        - Default PIN: "1234" (configurable in config.json)
        - Validates format and correctness
        
        Success Flow:
        1. Validate security status
        2. Validate PIN format (4 digits)
        3. Compare with correct PIN
        4. Transition from 'pin_entry' to 'menu' screen
        5. Return success response
        
        Failure Flow:
        1. Security violation → Cancel transaction, return to 'welcome'
        2. Invalid PIN format → Remain on 'pin_entry', show error
        3. Incorrect PIN → Remain on 'pin_entry', show error
        
        Request Body:
            JSON: {"pin": "1234"}
        
        Returns:
            JSON: {
                "success": true|false,
                "message": "PIN accepted" | "Incorrect PIN" | "Transaction cancelled: <violation>",
                "screen": "pin_entry|menu|welcome"
            }
        """
        transaction_controller = app.config['transaction_controller']
        
        data = request.get_json()
        pin = data.get('pin', '')
        
        result = transaction_controller.handle_pin_entry(pin)
        
        return jsonify(result)

    
    # ========================================================================
    # ROUTE 6: Select Withdrawal API
    # ========================================================================
    @app.route('/api/select_withdrawal', methods=['POST'])
    def api_select_withdrawal():
        """
        Navigate to withdrawal screen from main menu.
        
        Simple screen transition without security validation
        (security is validated during actual withdrawal processing).
        
        Flow:
        1. Transition from 'menu' to 'withdrawal' screen
        2. Clear withdrawal amount buffer
        3. Return success response
        
        Returns:
            JSON: {
                "success": true,
                "message": "Enter withdrawal amount",
                "screen": "withdrawal"
            }
        """
        atm_state = app.config['atm_state']
        
        result = atm_state.select_withdrawal()
        
        return jsonify(result)
    
    # ========================================================================
    # ROUTE 7: Process Withdrawal API
    # ========================================================================
    @app.route('/api/withdraw', methods=['POST'])
    def api_withdraw():
        """
        Process withdrawal transaction with comprehensive validation.
        
        Security Validation:
        - Checks security status before processing withdrawal
        - Cancels transaction if violation detected
        
        Amount Validation:
        - Must be positive number (> 0)
        - Must not exceed current balance
        - Must not exceed maximum withdrawal limit ($1000 default)
        
        Success Flow:
        1. Validate security status
        2. Validate amount format and value
        3. Check balance sufficiency
        4. Check withdrawal limit
        5. Deduct amount from balance
        6. Record transaction
        7. Transition to 'complete' screen
        8. Return success with new balance
        
        Failure Flow:
        1. Security violation → Cancel transaction, return to 'welcome'
        2. Invalid amount → Remain on 'withdrawal', show error
        3. Insufficient balance → Remain on 'withdrawal', show error
        4. Exceeds limit → Remain on 'withdrawal', show error
        
        Request Body:
            JSON: {"amount": 100.00}
        
        Returns:
            JSON: {
                "success": true|false,
                "message": "Withdrawal successful" | "<error_message>",
                "screen": "withdrawal|complete|welcome",
                "new_balance": 4900.00  (only on success)
            }
        """
        transaction_controller = app.config['transaction_controller']
        audio_system = app.config['audio_system']
        
        data = request.get_json()
        amount = data.get('amount', 0)
        
        result = transaction_controller.handle_withdrawal(amount)
        
        # Play completion message if transaction successful
        if result.get('success') and audio_system:
            audio_system.play_immediate('transaction_complete')
            logger.info("Completion message played: 'Thank you, your transaction is successfully completed'")
        
        return jsonify(result)

    
    # ========================================================================
    # ROUTE 8: Reset ATM API
    # ========================================================================
    @app.route('/api/reset', methods=['POST'])
    def api_reset():
        """
        Reset ATM to welcome screen.
        
        Used to:
        - Cancel current transaction
        - Return to initial state after transaction completion
        - Recover from error states
        
        Flow:
        1. Clear all transaction state
        2. Reset to 'welcome' screen
        3. Clear PIN input buffer
        4. Clear withdrawal amount
        5. Mark card as not inserted
        6. Return success response
        
        Note: Balance is NOT reset (persists across sessions)
        
        Returns:
            JSON: {
                "success": true,
                "message": "ATM reset to welcome screen",
                "screen": "welcome"
            }
        """
        atm_state = app.config['atm_state']
        
        result = atm_state.reset()
        
        return jsonify(result)

# ============================================================================
# SECTION 10: RESOURCE CLEANUP
# ============================================================================

def cleanup_resources(app):
    """
    Gracefully cleanup all system resources on shutdown.
    
    Cleanup Order:
    1. Audio system (stop background thread)
    2. Camera device (release hardware)
    3. OpenCV windows (destroy all)
    
    This function is called:
    - On normal application shutdown
    - On SIGINT (Ctrl+C)
    - On SIGTERM (kill command)
    - On unhandled exceptions
    
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
# SECTION 11: MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the ATM Security System.
    
    Execution Flow:
    1. Setup logging system
    2. Register signal handlers for graceful shutdown
    3. Create Flask application with all modules
    4. Start Flask server
    5. Handle shutdown signals
    6. Cleanup resources
    
    Signal Handling:
    - SIGINT (Ctrl+C): Graceful shutdown
    - SIGTERM (kill): Graceful shutdown
    
    Server Configuration:
    - Host: 0.0.0.0 (accessible from network)
    - Port: 5004 (configurable in config.json)
    - Debug: False (production mode)
    - Threaded: True (handle concurrent requests)
    
    Access URLs:
    - Local: http://127.0.0.1:5004
    - Network: http://<your-ip>:5004
    """
    # Setup logging first
    setup_logging(log_level=logging.INFO)
    
    logger.info("=" * 80)
    logger.info("ATM SECURITY SYSTEM - STARTING")
    logger.info("=" * 80)
    logger.info("System Components:")
    logger.info("  - Face Detection: Haar Cascade Classifier")
    logger.info("  - Mask Detection: PyTorch CNN (97.55% accuracy)")
    logger.info("  - Helmet Detection: PyTorch CNN (99.24% accuracy)")
    logger.info("  - Person Detection: PyTorch CNN (87.50% accuracy)")
    logger.info("  - Security Manager: Access Control Logic")
    logger.info("  - Audio Alerts: Voice Warnings (4 violation types)")
    logger.info("  - ATM Interface: Web-based Transaction System")
    logger.info("  - Video Streaming: MJPEG with Security Overlays")
    logger.info("=" * 80)
    
    app = None
    
    def signal_handler(signum, frame):
        """
        Handle shutdown signals gracefully.
        
        Called when:
        - User presses Ctrl+C (SIGINT)
        - System sends kill signal (SIGTERM)
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if app:
            cleanup_resources(app)
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create Flask application with all modules
        app = create_app()
        
        # Get configuration
        config = app.config['ATM_CONFIG']
        port = config.get('flask_port', 5004)
        
        logger.info("=" * 80)
        logger.info(f"Starting Flask server on port {port}")
        logger.info(f"Access URLs:")
        logger.info(f"  - Local:   http://127.0.0.1:{port}")
        logger.info(f"  - Network: http://<your-ip>:{port}")
        logger.info("=" * 80)
        logger.info("System ready. Press Ctrl+C to stop.")
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
        logger.info("ATM SECURITY SYSTEM - SHUTDOWN COMPLETE")
        logger.info("=" * 80)
        sys.exit(0)

# ============================================================================
# SECTION 12: SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Execute main function when script is run directly.
    
    Usage:
        python app_complete_commented.py
    
    Requirements:
        - config.json must exist in same directory
        - Camera must be connected and accessible
        - Audio files must be present in audio/ directory
        - Trained models must be present in models/ directory
        - Templates and static files must be present
    
    Configuration:
        Edit config.json to customize:
        - Camera settings (index, resolution, FPS)
        - Detection thresholds (mask, helmet)
        - Audio settings (enabled, rate limiting)
        - ATM parameters (balance, PIN, limits)
        - Server port
    """
    main()

# ============================================================================
# END OF FILE
# ============================================================================
"""
SECURITY VIOLATION TYPES:
========================

1. NO_FACE_DETECTED
   - Trigger: face_count == 0
   - Meaning: No person visible in camera
   - Action: Deny access, no audio alert

2. MULTIPLE_PEOPLE
   - Trigger: face_count > 1 OR person_detector detects multiple_person
   - Meaning: More than one person at ATM (fraud attempt)
   - Action: Deny access, play "multiple_people.mp3"

3. MASK_DETECTED
   - Trigger: mask_detector confidence > 0.5 (with_mask class)
   - Meaning: Face mask covering mouth/nose
   - Action: Deny access, play "mask_detected.mp3"

4. HELMET_DETECTED
   - Trigger: helmet_detector confidence > 0.4 (with_helmet class)
   - Meaning: Helmet or head covering detected
   - Action: Deny access, play "helmet_detected.mp3"

5. MASK_AND_HELMET_DETECTED
   - Trigger: Both mask AND helmet detected simultaneously
   - Meaning: Multiple face coverings (high-risk fraud attempt)
   - Action: Deny access, play "mask_and_helmet.mp3"

ACCESS GRANTED:
==============
- Trigger: face_count == 1 AND no_mask AND no_helmet AND single_person
- Meaning: Single person with clear, visible face
- Action: Grant access, allow transaction to proceed

MODEL DETAILS:
=============

1. MASK DETECTOR (mask_detector.pth)
   - Architecture: 3 Conv blocks (32→64→128 filters) + FC layers
   - Input: 128x128 RGB image
   - Output: 2 classes (with_mask, without_mask)
   - Training: 6,042 samples, 20 epochs
   - Validation Accuracy: 97.55%
   - Threshold: 0.5 (configurable)

2. HELMET DETECTOR (helmet_detector.pth)
   - Architecture: 3 Conv blocks (32→64→128 filters) + FC layers
   - Input: 128x128 RGB image
   - Output: 2 classes (with_helmet, without_helmet)
   - Training: 3,140 samples, 20 epochs
   - Validation Accuracy: 99.24%
   - Threshold: 0.4 (configurable)

3. PERSON DETECTOR (person_detector.pth)
   - Architecture: 3 Conv blocks (32→64→128 filters) + FC layers
   - Input: 128x128 RGB image
   - Output: 2 classes (multiple_person, single_person)
   - Training: 32 samples, 20 epochs
   - Validation Accuracy: 87.50%
   - Threshold: 0.5 (configurable)

VIDEO PROCESSING PIPELINE:
=========================

Frame Capture → Face Detection (Haar Cascade)
    ↓
ROI Extraction → Preprocessing (CLAHE)
    ↓
Parallel CNN Inference:
    - Mask Detection
    - Helmet Detection
    - Person Detection (full frame)
    ↓
Security Status Update
    ↓
Visual Overlays (boxes, text, violations)
    ↓
Audio Alert Queue (if violation)
    ↓
MJPEG Encoding → Stream to Browser

PERFORMANCE METRICS:
===================
- Frame Processing: <100ms per frame
- Video Streaming: 15+ FPS
- API Response: <100ms
- Memory Usage: <500MB
- CPU Usage: <50% (dual-core)
- Continuous Operation: 8+ hours tested

For more information, see:
- README.md: User documentation and setup
- DEPLOYMENT.md: Production deployment guide
- MODEL_TRAINING_SUMMARY.md: Model training details
- PROJECT_ANALYSIS_REPORT.md: Comprehensive project analysis
"""
