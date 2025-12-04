"""
Video Stream Handler Module

This module provides video streaming functionality with security overlays for the web interface.
It generates MJPEG streams with visual indicators for face detection and security status.
"""

import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class VideoStreamHandler:
    """
    Handles video streaming with security overlays for web display.
    
    This class provides methods to:
    - Generate MJPEG video streams for web browsers
    - Draw visual overlays on video frames (face boxes, status text)
    - Add color-coded indicators based on security status
    """
    
    def __init__(self, target_fps=15):
        """
        Initialize the VideoStreamHandler.
        
        Args:
            target_fps: Target frames per second for video streaming (default: 15)
        """
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time = 0
        logger.info(f"VideoStreamHandler initialized with target FPS: {target_fps}")
    
    def generate_frames(self, camera, face_detector, mask_detector, helmet_detector, security_manager, person_detector=None):
        """
        Generator that yields MJPEG frames for video streaming.
        
        This generator continuously captures frames from the camera, processes them
        for face/mask/helmet/person detection, updates security status, and yields encoded
        frames with visual overlays.
        
        Args:
            camera: OpenCV VideoCapture object
            face_detector: FaceDetector instance
            mask_detector: MaskDetector instance
            helmet_detector: HelmetDetector instance
            security_manager: SecurityStatusManager instance
            person_detector: PersonDetector instance (optional)
        
        Yields:
            bytes: MJPEG frame data in multipart format
        """
        consecutive_detection_errors = 0
        max_consecutive_errors = 10
        
        while True:
            frame = None
            face_roi = None
            preprocessed_face = None
            buffer = None
            
            try:
                success, frame = camera.read()
                
                if not success:
                    logger.warning("Failed to capture frame from camera")
                    break
                # Detect faces in the frame
                faces = face_detector.detect_faces(frame)
                face_count = len(faces)
                
                # Initialize detection results
                has_mask = False
                has_helmet = False
                mask_confidence = 0.0
                helmet_confidence = 0.0
                is_multiple_persons = False
                person_confidence = 0.0
                
                # Check for multiple persons if detector is available
                if person_detector is not None:
                    try:
                        is_multiple_persons, person_confidence = person_detector.detect_multiple_persons(frame)
                    except Exception as e:
                        logger.error(f"Person detection error: {e}")
                
                # If exactly one face detected, perform mask and helmet detection
                if face_count == 1:
                    face_rect = faces[0]
                    face_roi = face_detector.extract_face_roi(frame, face_rect)
                    
                    if face_roi is not None:
                        # Preprocess face for detection
                        preprocessed_face = face_detector.preprocess_face(face_roi)
                        
                        if preprocessed_face is not None:
                            try:
                                # Perform mask detection
                                has_mask, mask_confidence = mask_detector.detect_mask(preprocessed_face)
                                
                                # Perform helmet detection
                                has_helmet, helmet_confidence = helmet_detector.detect_helmet(preprocessed_face)
                                
                                # Reset error counter on successful detection
                                consecutive_detection_errors = 0
                                
                            except Exception as e:
                                logger.error(f"Detection error: {e}")
                                consecutive_detection_errors += 1
                                
                                # Log persistent detection errors
                                if consecutive_detection_errors >= max_consecutive_errors:
                                    logger.warning(
                                        f"Persistent detection errors: {consecutive_detection_errors} consecutive failures"
                                    )
                                
                                # Default to access denied on detection failure (fail-safe)
                                # Skip this frame and continue with next
                                security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0, 'person': 0.0}, False)
                                continue
                
                # FALLBACK: If no face detected, check for helmet on full frame
                # This handles cases where helmet completely covers the face
                elif face_count == 0:
                    try:
                        # Strategy 1: Check full frame
                        frame_resized = cv2.resize(frame, (128, 128))
                        has_helmet_fullframe, helmet_confidence_fullframe = helmet_detector.detect_helmet(frame_resized)
                        
                        # Strategy 2: Check upper 60% of frame (where head typically is)
                        height = frame.shape[0]
                        upper_frame = frame[0:int(height*0.6), :]
                        upper_resized = cv2.resize(upper_frame, (128, 128))
                        has_helmet_upper, helmet_confidence_upper = helmet_detector.detect_helmet(upper_resized)
                        
                        # Strategy 3: Check center region (typical head position)
                        h, w = frame.shape[:2]
                        center_y1, center_y2 = int(h*0.1), int(h*0.7)
                        center_x1, center_x2 = int(w*0.2), int(w*0.8)
                        center_frame = frame[center_y1:center_y2, center_x1:center_x2]
                        if center_frame.size > 0:
                            center_resized = cv2.resize(center_frame, (128, 128))
                            has_helmet_center, helmet_confidence_center = helmet_detector.detect_helmet(center_resized)
                        else:
                            has_helmet_center, helmet_confidence_center = False, 0.0
                        
                        # Use the highest confidence detection
                        max_confidence = max(helmet_confidence_fullframe, helmet_confidence_upper, helmet_confidence_center)
                        # Lower threshold to 0.20 to catch motorcycle helmets not in training data
                        has_helmet_detected = (helmet_confidence_fullframe > 0.20 or 
                                             helmet_confidence_upper > 0.20 or 
                                             helmet_confidence_center > 0.20)
                        
                        # Log all detection results for debugging
                        logger.info(f"Helmet detection (no face): full={helmet_confidence_fullframe:.2f}, "
                                  f"upper={helmet_confidence_upper:.2f}, center={helmet_confidence_center:.2f}, "
                                  f"max={max_confidence:.2f}")
                        
                        # If helmet detected with reasonable confidence, update status
                        # Lowered threshold to 0.20 for motorcycle helmets
                        if has_helmet_detected and max_confidence > 0.20:
                            logger.info(f"✓ HELMET DETECTED (no face): confidence={max_confidence:.2f}")
                            # Set face_count to 1 to indicate presence, and mark helmet detected
                            face_count = 1
                            has_helmet = True
                            helmet_confidence = max_confidence
                        else:
                            logger.info(f"✗ No helmet detected: max_confidence={max_confidence:.2f} (threshold=0.20)")
                            
                    except Exception as e:
                        logger.error(f"Full frame helmet detection error: {e}", exc_info=True)
                
                # Update security status
                confidences = {
                    'mask': mask_confidence,
                    'helmet': helmet_confidence,
                    'person': person_confidence
                }
                security_manager.update_status(face_count, has_mask, has_helmet, confidences, is_multiple_persons)
                
                # Get current security status
                security_status = security_manager.get_status()
                
                # Draw overlays on the frame
                self._draw_overlays(frame, faces, security_status)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                
                if not ret:
                    logger.warning("Failed to encode frame as JPEG")
                    continue
                
                # Convert to bytes
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format for MJPEG streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # CPU optimization: Control frame rate to reduce CPU usage
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)
                self.last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                consecutive_detection_errors += 1
                
                # Log persistent errors
                if consecutive_detection_errors >= max_consecutive_errors:
                    logger.warning(
                        f"Persistent frame processing errors: {consecutive_detection_errors} consecutive failures"
                    )
                
                # Default to access denied on processing failure (fail-safe)
                security_manager.update_status(0, False, False, {'mask': 0.0, 'helmet': 0.0, 'person': 0.0}, False)
                continue
            
            finally:
                # Release frame memory after processing
                # This ensures memory is freed even if an exception occurs
                if frame is not None:
                    del frame
                if face_roi is not None:
                    del face_roi
                if preprocessed_face is not None:
                    del preprocessed_face
                if buffer is not None:
                    del buffer
    
    def _draw_overlays(self, frame, faces, security_status):
        """
        Add visual indicators to the frame.
        
        Draws face boxes and security status text on the frame based on
        the current security state.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on (modified in place)
            faces (list): List of face rectangles [(x, y, w, h), ...]
            security_status (dict): Current security status from SecurityStatusManager
        """
        # Draw face boxes
        self._draw_face_boxes(frame, faces, security_status)
        
        # Draw status text
        self._draw_status_text(frame, security_status)
    
    def _draw_face_boxes(self, frame, faces, security_status):
        """
        Draw rectangles around detected faces.
        
        Uses color coding:
        - Green: Access granted (clear face detected)
        - Red: Access denied (violation detected)
        
        Args:
            frame (numpy.ndarray): Video frame to draw on (modified in place)
            faces (list): List of face rectangles [(x, y, w, h), ...]
            security_status (dict): Current security status
        """
        # Determine box color based on security status
        if security_status['access_granted']:
            color = (0, 255, 0)  # Green for access granted
        else:
            color = (0, 0, 255)  # Red for access denied
        
        # If faces detected, draw boxes around them
        if len(faces) > 0:
            # Draw rectangle around each detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw detection indicators if applicable
                if security_status['has_mask']:
                    # Draw "MASK" label above face box
                    label = f"MASK: {security_status['mask_confidence']:.2f}"
                    cv2.putText(frame, label, (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if security_status['has_helmet']:
                    # Draw "HELMET" label above face box
                    label = f"HELMET: {security_status['helmet_confidence']:.2f}"
                    y_offset = -10 if security_status['has_mask'] else -30
                    cv2.putText(frame, label, (x, y + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # If no faces but helmet detected (full-frame detection), draw center box
        elif security_status['has_helmet'] and not security_status['access_granted']:
            # Draw a box in the center-upper area where head typically is
            h, w = frame.shape[:2]
            box_w = int(w * 0.4)  # 40% of frame width
            box_h = int(h * 0.5)  # 50% of frame height
            x = int(w * 0.3)  # Center horizontally
            y = int(h * 0.1)  # Upper portion
            
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
            
            # Draw "HELMET" label
            label = f"HELMET: {security_status['helmet_confidence']:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def generate_error_frame(self, error_message):
        """
        Generate an error frame with a message.
        
        Args:
            error_message (str): Error message to display
        
        Returns:
            bytes: MJPEG frame data with error message
        """
        # Create a black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add error message
        cv2.putText(frame, "ERROR", (250, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, error_message, (50, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if ret:
            frame_bytes = buffer.tobytes()
            return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            return b''
    
    def _draw_status_text(self, frame, security_status):
        """
        Display security status text on the frame.
        
        Shows the current security state and any violation messages
        in the top-left corner of the frame.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on (modified in place)
            security_status (dict): Current security status
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Determine status text and color
        if security_status['access_granted']:
            status_text = "ACCESS GRANTED"
            text_color = (0, 255, 0)  # Green
        else:
            status_text = "ACCESS DENIED"
            text_color = (0, 0, 255)  # Red
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw status text
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Draw face count
        face_text = f"Faces: {security_status['face_count']}"
        cv2.putText(frame, face_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw violation message if access denied
        if not security_status['access_granted'] and security_status['violation_type']:
            violation_text = security_status['violation_type'].replace('_', ' ').upper()
            cv2.putText(frame, violation_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
