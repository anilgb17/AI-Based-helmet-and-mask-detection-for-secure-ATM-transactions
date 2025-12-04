"""
Code Snippets for Integrating Welcome and Completion Audio
============================================================

Copy these code snippets into app_complete_commented.py at the specified locations.
"""

# ============================================================================
# SNIPPET 1: Welcome Message - Play when video stream starts
# ============================================================================
# Location: In the video() route, inside the generate() function
# Add after: while True: try:

"""
# Track if welcome message has been played
welcome_played = False

# Inside the while True loop, after successful frame capture:
if not welcome_played and audio_system:
    audio_system.play_immediate('welcome')
    welcome_played = True
    logger.info("✓ Welcome message played")
"""

# ============================================================================
# SNIPPET 2: Completion Message - Play after successful withdrawal
# ============================================================================
# Location: In the api_withdraw() route
# Add after: result = transaction_controller.handle_withdrawal(amount)

"""
# Play completion message if transaction successful
if result.get('success') and audio_system:
    audio_system.play_immediate('transaction_complete')
    logger.info("✓ Transaction completion message played")
"""

# ============================================================================
# COMPLETE UPDATED video() FUNCTION
# ============================================================================

def video_with_welcome():
    """
    Updated video() function with welcome message integration
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
        """Generator function with welcome message"""
        nonlocal camera
        last_reconnect_attempt = 0
        reconnect_interval = 5
        welcome_played = False  # NEW: Track welcome message
        
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
                
                # NEW: Play welcome message on first successful frame
                if not welcome_played and audio_system:
                    audio_system.play_immediate('welcome')
                    welcome_played = True
                    logger.info("✓ Welcome message played")
                
                # Generate frames with detection and overlays
                for frame_data in video_handler.generate_frames(
                    camera, face_detector, mask_detector, helmet_detector, 
                    security_manager, person_detector
                ):
                    # Queue audio alert if violation detected
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

# ============================================================================
# COMPLETE UPDATED api_withdraw() FUNCTION
# ============================================================================

def api_withdraw_with_completion():
    """
    Updated api_withdraw() function with completion message integration
    """
    transaction_controller = app.config['transaction_controller']
    audio_system = app.config['audio_system']
    
    data = request.get_json()
    amount = data.get('amount', 0)
    
    result = transaction_controller.handle_withdrawal(amount)
    
    # NEW: Play completion message if transaction successful
    if result.get('success') and audio_system:
        audio_system.play_immediate('transaction_complete')
        logger.info("✓ Transaction completion message played")
    
    return jsonify(result)

# ============================================================================
# ALTERNATIVE: Welcome on Card Insertion
# ============================================================================

def api_insert_card_with_welcome():
    """
    Alternative: Play welcome message when card is inserted
    """
    transaction_controller = app.config['transaction_controller']
    audio_system = app.config['audio_system']
    
    result = transaction_controller.handle_card_insertion()
    
    # Play welcome message on successful card insertion
    if result.get('success') and audio_system:
        audio_system.play_immediate('welcome')
        logger.info("✓ Welcome message played on card insertion")
    
    return jsonify(result)

# ============================================================================
# BONUS: Goodbye Message on Reset
# ============================================================================

def api_reset_with_goodbye():
    """
    Bonus: Play goodbye message when ATM resets
    """
    atm_state = app.config['atm_state']
    audio_system = app.config['audio_system']
    
    result = atm_state.reset()
    
    # Play goodbye message (if you have goodbye.mp3)
    if audio_system:
        # Check if goodbye audio exists
        goodbye_path = os.path.join(audio_system.audio_dir, 'goodbye.mp3')
        if os.path.exists(goodbye_path):
            audio_system.play_immediate('goodbye')
            logger.info("✓ Goodbye message played")
    
    return jsonify(result)

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
HOW TO USE THESE SNIPPETS:

1. WELCOME MESSAGE (Option A - When Video Starts):
   - Find the video() function in app_complete_commented.py
   - Add the welcome_played variable and check inside generate()
   - See SNIPPET 1 above

2. WELCOME MESSAGE (Option B - When Card Inserted):
   - Find the api_insert_card() function
   - Add the welcome message call after successful insertion
   - See api_insert_card_with_welcome() above

3. COMPLETION MESSAGE:
   - Find the api_withdraw() function
   - Add the completion message call after successful withdrawal
   - See SNIPPET 2 above

4. RESTART APPLICATION:
   python run_atm_system.py

5. TEST:
   - Open http://127.0.0.1:5004
   - Listen for welcome message
   - Complete a transaction
   - Listen for completion message
"""
