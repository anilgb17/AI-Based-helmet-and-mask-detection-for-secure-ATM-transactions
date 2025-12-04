"""Audio Alert System for playing security violation warnings."""

import os
import time
import threading
import queue
import logging
import pygame

logger = logging.getLogger(__name__)


class AudioAlertSystem:
    """Manages audio alerts for security violations using background thread processing."""
    
    def __init__(self, audio_dir='audio'):
        """
        Initialize the audio alert system.
        
        Args:
            audio_dir: Directory containing MP3 audio files
            
        Raises:
            FileNotFoundError: If audio directory or required files don't exist
        """
        self.audio_dir = audio_dir
        self.message_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.last_alert_time = 0
        self.rate_limit_seconds = 8
        self.mixer_initialized = False
        self.audio_enabled = True
        
        # Initialize pygame mixer with error handling
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
            logger.info("Pygame mixer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            self.audio_enabled = False
        
        # Define audio file mapping
        self.audio_files = {
            # Security violation alerts
            'multiple_people': os.path.join(audio_dir, 'multiple_people.mp3'),
            'mask_detected': os.path.join(audio_dir, 'mask_detected.mp3'),
            'helmet_detected': os.path.join(audio_dir, 'helmet_detected.mp3'),
            'mask_and_helmet_detected': os.path.join(audio_dir, 'mask_and_helmet.mp3'),
            # Welcome and completion messages (NEW)
            'welcome': os.path.join(audio_dir, 'welcome.mp3'),
            'transaction_complete': os.path.join(audio_dir, 'transaction_complete.mp3')
        }
        
        # Validate audio files exist
        try:
            self._validate_audio_files()
            logger.info("All audio files validated successfully")
        except FileNotFoundError as e:
            logger.warning(f"Audio file validation failed: {e}")
            self.audio_enabled = False
        
        # Start the audio worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio worker thread started")
    
    def _validate_audio_files(self):
        """
        Verify all required MP3 files exist and are accessible.
        
        Raises:
            FileNotFoundError: If any required audio file is missing
        """
        if not os.path.exists(self.audio_dir):
            error_msg = f"Audio directory not found: {self.audio_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Required files (security violations)
        required_files = ['multiple_people', 'mask_detected', 'helmet_detected', 'mask_and_helmet_detected']
        # Optional files (welcome and completion)
        optional_files = ['welcome', 'transaction_complete']
        
        missing_required = []
        inaccessible_files = []
        
        for violation_type, file_path in self.audio_files.items():
            if not os.path.exists(file_path):
                if violation_type in required_files:
                    missing_required.append(file_path)
                    logger.warning(f"Required audio file not found: {file_path}")
                else:
                    logger.info(f"Optional audio file not found: {file_path}")
            elif not os.access(file_path, os.R_OK):
                inaccessible_files.append(file_path)
                logger.warning(f"Audio file not accessible: {file_path}")
        
        if missing_required:
            error_msg = f"Missing required audio files: {', '.join(missing_required)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if inaccessible_files:
            error_msg = f"Inaccessible audio files: {', '.join(inaccessible_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Log available optional files
        for optional in optional_files:
            file_path = self.audio_files.get(optional)
            if file_path and os.path.exists(file_path):
                logger.info(f"Optional audio file available: {optional}")
    
    def queue_alert(self, violation_type):
        """
        Add an alert message to the queue for playback.
        
        Args:
            violation_type: Type of security violation (e.g., 'mask_detected')
        """
        if violation_type in self.audio_files:
            self.message_queue.put(violation_type)
    
    def play_immediate(self, message_type):
        """
        Play audio message immediately without rate limiting.
        Used for welcome and completion messages.
        
        Args:
            message_type: Type of message ('welcome', 'transaction_complete')
        """
        if not self.audio_enabled:
            logger.debug("Audio disabled, skipping immediate playback")
            return
        
        audio_file = self.audio_files.get(message_type)
        if audio_file:
            # Play in a separate thread to avoid blocking
            threading.Thread(target=self._play_audio, args=(audio_file,), daemon=True).start()
            logger.info(f"Playing immediate audio: {message_type}")
    
    def play_system_message(self, message_type):
        """
        Alias for play_immediate() for backward compatibility.
        Play system message (welcome, transaction_complete) without rate limiting.
        
        Args:
            message_type: Type of message ('welcome', 'transaction_complete')
        """
        self.play_immediate(message_type)
    
    def _audio_worker(self):
        """
        Background thread loop that processes queued audio messages.
        Implements rate limiting to prevent audio spam.
        """
        while self.running:
            try:
                # Wait for a message with timeout to allow checking running flag
                violation_type = self.message_queue.get(timeout=1.0)
                
                # Skip if audio is disabled
                if not self.audio_enabled:
                    logger.debug("Audio disabled, skipping alert")
                    self.message_queue.task_done()
                    continue
                
                # Check rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_alert_time
                
                if time_since_last >= self.rate_limit_seconds:
                    # Get audio file path and play
                    audio_file = self._get_audio_file(violation_type)
                    if audio_file:
                        self._play_audio(audio_file)
                        self.last_alert_time = current_time
                
                self.message_queue.task_done()
                
            except queue.Empty:
                # No message in queue, sleep for 1 second to reduce CPU usage when idle
                time.sleep(1)
            except Exception as e:
                # Log error but continue operation
                logger.error(f"Audio worker error: {e}")
                # Sleep to prevent tight error loop
                time.sleep(1)
    
    def _get_audio_file(self, violation_type):
        """
        Map violation type to MP3 file path.
        
        Args:
            violation_type: Type of security violation
            
        Returns:
            File path to corresponding MP3 file, or None if not found
        """
        return self.audio_files.get(violation_type)
    
    def _play_audio(self, file_path):
        """
        Play MP3 file using pygame mixer with error handling and reinitialization.
        
        Args:
            file_path: Path to MP3 file to play
        """
        try:
            # Verify file still exists and is accessible
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return
            
            if not os.access(file_path, os.R_OK):
                logger.error(f"Audio file not accessible: {file_path}")
                return
            
            # Try to play the audio
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            logger.debug(f"Successfully played audio: {file_path}")
                
        except pygame.error as e:
            logger.error(f"Pygame error playing audio file {file_path}: {e}")
            
            # Attempt to reinitialize pygame mixer
            try:
                logger.info("Attempting to reinitialize pygame mixer...")
                pygame.mixer.quit()
                time.sleep(0.5)
                pygame.mixer.init()
                self.mixer_initialized = True
                logger.info("Pygame mixer reinitialized successfully")
                
                # Retry playing the audio once
                try:
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    logger.info(f"Successfully played audio after reinitialization: {file_path}")
                except Exception as retry_error:
                    logger.error(f"Failed to play audio after reinitialization: {retry_error}")
                    self.audio_enabled = False
                    
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize pygame mixer: {reinit_error}")
                self.mixer_initialized = False
                self.audio_enabled = False
                
        except Exception as e:
            logger.error(f"Unexpected error playing audio file {file_path}: {e}")
            self.audio_enabled = False
    
    def shutdown(self):
        """Gracefully stop the audio worker thread and cleanup resources."""
        logger.info("Shutting down audio alert system...")
        self.running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            logger.info("Audio worker thread stopped")
        
        # Stop any playing audio
        try:
            if self.mixer_initialized:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                logger.info("Pygame mixer shutdown complete")
        except Exception as e:
            logger.error(f"Error during pygame mixer shutdown: {e}")
