"""
Test Helmet Detection with Audio
Checks if helmet detection triggers audio alert
"""
import cv2
import time
from detection.face_detector import FaceDetector
from detection.helmet_detector_pytorch import HelmetDetectorPyTorch as HelmetDetector
from security.security_status_manager import SecurityStatusManager
from audio.audio_alert_system import AudioAlertSystem

print("=" * 80)
print("HELMET DETECTION + AUDIO TEST")
print("=" * 80)

# Initialize components
print("\n[1/5] Initializing components...")
face_detector = FaceDetector()
helmet_detector = HelmetDetector(threshold=0.4)
security_manager = SecurityStatusManager()
audio_system = AudioAlertSystem('audio')
print("✓ All components initialized")

# Open camera
print("\n[2/5] Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Failed to open camera")
    exit(1)
print("✓ Camera opened")

# Capture frame
print("\n[3/5] Capturing frame...")
ret, frame = cap.read()
if not ret:
    print("✗ Failed to capture frame")
    cap.release()
    exit(1)
print(f"✓ Frame captured: {frame.shape[1]}x{frame.shape[0]}")

# Detect faces
print("\n[4/5] Running detection...")
faces = face_detector.detect_faces(frame)
face_count = len(faces)
print(f"  Faces detected: {face_count}")

# Initialize detection results
has_mask = False
has_helmet = False
mask_confidence = 0.0
helmet_confidence = 0.0

# Check for helmet
if face_count == 1:
    print("\n  [Face Region Detection]")
    face_rect = faces[0]
    face_roi = face_detector.extract_face_roi(frame, face_rect)
    if face_roi is not None:
        preprocessed_face = face_detector.preprocess_face(face_roi)
        if preprocessed_face is not None:
            has_helmet, helmet_confidence = helmet_detector.detect_helmet(preprocessed_face)
            print(f"    Helmet detected: {has_helmet}")
            print(f"    Confidence: {helmet_confidence:.4f}")

elif face_count == 0:
    print("\n  [Full Frame Detection - Fallback]")
    # Strategy 1: Full frame
    frame_resized = cv2.resize(frame, (128, 128))
    has_helmet_full, conf_full = helmet_detector.detect_helmet(frame_resized)
    print(f"    Full frame: {has_helmet_full}, confidence={conf_full:.4f}")
    
    # Strategy 2: Upper 60%
    height = frame.shape[0]
    upper_frame = frame[0:int(height*0.6), :]
    upper_resized = cv2.resize(upper_frame, (128, 128))
    has_helmet_upper, conf_upper = helmet_detector.detect_helmet(upper_resized)
    print(f"    Upper 60%: {has_helmet_upper}, confidence={conf_upper:.4f}")
    
    # Strategy 3: Center region
    h, w = frame.shape[:2]
    center_y1, center_y2 = int(h*0.1), int(h*0.7)
    center_x1, center_x2 = int(w*0.2), int(w*0.8)
    center_frame = frame[center_y1:center_y2, center_x1:center_x2]
    center_resized = cv2.resize(center_frame, (128, 128))
    has_helmet_center, conf_center = helmet_detector.detect_helmet(center_resized)
    print(f"    Center region: {has_helmet_center}, confidence={conf_center:.4f}")
    
    # Use highest confidence
    max_confidence = max(conf_full, conf_upper, conf_center)
    print(f"\n    Maximum confidence: {max_confidence:.4f}")
    print(f"    Threshold: 0.20")
    
    if max_confidence > 0.20:
        print(f"    ✓ HELMET DETECTED (confidence > 0.20)")
        face_count = 1
        has_helmet = True
        helmet_confidence = max_confidence
    else:
        print(f"    ✗ No helmet (confidence < 0.20)")

# Update security status
print("\n[5/5] Updating security status...")
confidences = {
    'mask': mask_confidence,
    'helmet': helmet_confidence,
    'person': 0.0
}
security_manager.update_status(face_count, has_mask, has_helmet, confidences, False)

# Get security status
security_status = security_manager.get_status()
print(f"\n  Access granted: {security_status['access_granted']}")
print(f"  Violation type: {security_status['violation_type']}")
print(f"  Face count: {security_status['face_count']}")
print(f"  Has helmet: {security_status['has_helmet']}")
print(f"  Helmet confidence: {security_status['helmet_confidence']:.4f}")

# Queue audio alert if violation
print("\n[AUDIO] Checking for audio alert...")
if not security_status['access_granted'] and security_status['violation_type']:
    print(f"  Violation detected: {security_status['violation_type']}")
    print(f"  Queuing audio alert...")
    audio_system.queue_alert(security_status['violation_type'])
    print(f"  ✓ Audio alert queued")
    print(f"\n  Waiting 5 seconds for audio to play...")
    print(f"  Listen for: 'Helmet detected. Please remove helmet...'")
    time.sleep(5)
    print(f"  ✓ Audio should have played")
else:
    print(f"  No violation - no audio alert")

# Cleanup
cap.release()
audio_system.shutdown()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

if security_status['violation_type'] == 'helmet_detected':
    print("\n✓ Helmet detection working correctly")
    print("✓ Audio alert queued")
    print("\nDid you hear the helmet detection audio?")
    print("  YES → System working perfectly!")
    print("  NO  → Check volume or audio file")
else:
    print("\n✗ Helmet not detected")
    print("  Possible reasons:")
    print("  - Helmet confidence < 0.20")
    print("  - Model not trained on this helmet type")
    print("  - Need to retrain model")
