"""
Simple Helmet Detection Test (Console Only)
No GUI required - just shows confidence values
"""
import cv2
from detection.helmet_detector_pytorch import HelmetDetectorPyTorch

print("=" * 70)
print("HELMET DETECTION TEST (Console Only)")
print("=" * 70)

# Initialize detector
print("\nInitializing helmet detector...")
detector = HelmetDetectorPyTorch(threshold=0.35)
print("✓ Detector ready")

# Open camera
print("\nOpening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ ERROR: Cannot open camera")
    exit(1)
print("✓ Camera opened")

# Capture frame
print("\nCapturing frame...")
ret, frame = cap.read()
if not ret:
    print("✗ ERROR: Cannot capture frame")
    cap.release()
    exit(1)
print(f"✓ Frame captured: {frame.shape[1]}x{frame.shape[0]}")

# Test detection
print("\n" + "=" * 70)
print("TESTING HELMET DETECTION")
print("=" * 70)

# Strategy 1: Full frame
frame_resized = cv2.resize(frame, (128, 128))
has_helmet_full, conf_full = detector.detect_helmet(frame_resized)
print(f"\n[Strategy 1] Full Frame")
print(f"  Detected: {has_helmet_full}")
print(f"  Confidence: {conf_full:.4f}")

# Strategy 2: Upper 60%
height = frame.shape[0]
upper_frame = frame[0:int(height*0.6), :]
upper_resized = cv2.resize(upper_frame, (128, 128))
has_helmet_upper, conf_upper = detector.detect_helmet(upper_resized)
print(f"\n[Strategy 2] Upper 60%")
print(f"  Detected: {has_helmet_upper}")
print(f"  Confidence: {conf_upper:.4f}")

# Strategy 3: Center region
h, w = frame.shape[:2]
center_y1, center_y2 = int(h*0.1), int(h*0.7)
center_x1, center_x2 = int(w*0.2), int(w*0.8)
center_frame = frame[center_y1:center_y2, center_x1:center_x2]
center_resized = cv2.resize(center_frame, (128, 128))
has_helmet_center, conf_center = detector.detect_helmet(center_resized)
print(f"\n[Strategy 3] Center Region")
print(f"  Detected: {has_helmet_center}")
print(f"  Confidence: {conf_center:.4f}")

# Final decision
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
max_confidence = max(conf_full, conf_upper, conf_center)
threshold = 0.20  # Lowered to detect motorcycle helmets
helmet_detected = max_confidence > threshold

print(f"\nMaximum Confidence: {max_confidence:.4f}")
print(f"Threshold: {threshold}")
print(f"\nDecision: ", end="")
if helmet_detected:
    print("✓ HELMET DETECTED")
    print("  → System will show: ACCESS DENIED - HELMET DETECTED")
else:
    print("✗ NO HELMET DETECTED")
    print("  → System will show: NO FACE DETECTED (if no face found)")

print("\n" + "=" * 70)

# Interpretation
print("\nINTERPRETATION:")
if max_confidence > 0.50:
    print("  Strong detection - helmet clearly visible")
elif max_confidence > 0.35:
    print("  Good detection - helmet detected with reasonable confidence")
elif max_confidence > 0.20:
    print("  Weak detection - helmet detected but model not trained on this type")
    print("  (Threshold lowered to 0.20 to catch motorcycle helmets)")
elif max_confidence > 0.10:
    print("  Very weak - helmet might be present but model doesn't recognize it")
else:
    print("  No detection - helmet not recognized by model at all")

# Recommendations
if not helmet_detected and max_confidence > 0.10:
    print("\nRECOMMENDATION:")
    print("  Model was not trained on your helmet type (motorcycle helmet)")
    print("  Options:")
    print("  1. Retrain model with motorcycle helmet images")
    print("  2. Lower threshold further (may increase false positives)")
    print("  3. Use alternative detection method (YOLO)")
elif helmet_detected and max_confidence < 0.30:
    print("\nNOTE:")
    print("  Helmet detected with low confidence")
    print("  This means model wasn't trained on this helmet type")
    print("  Consider retraining with motorcycle helmet images for better accuracy")

# Cleanup
cap.release()
print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
