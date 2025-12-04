"""
Quick Helmet Detection Test Script
Tests helmet detection on current camera frame
"""
import cv2
import sys
from detection.helmet_detector_pytorch import HelmetDetectorPyTorch

def test_helmet_detection():
    """Test helmet detection with current camera view"""
    print("=" * 60)
    print("HELMET DETECTION TEST")
    print("=" * 60)
    
    # Initialize detector
    print("\n[1/4] Initializing helmet detector...")
    try:
        detector = HelmetDetectorPyTorch(threshold=0.35)
        print("✓ Helmet detector initialized")
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        return False
    
    # Open camera
    print("\n[2/4] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return False
    print("✓ Camera opened")
    
    # Capture frame
    print("\n[3/4] Capturing frame...")
    ret, frame = cap.read()
    if not ret or frame is None:
        print("✗ Failed to capture frame")
        cap.release()
        return False
    print(f"✓ Frame captured: {frame.shape}")
    
    # Test detection with multiple strategies
    print("\n[4/4] Testing helmet detection...")
    print("-" * 60)
    
    # Strategy 1: Full frame
    frame_resized = cv2.resize(frame, (128, 128))
    has_helmet_full, conf_full = detector.detect_helmet(frame_resized)
    print(f"Strategy 1 (Full Frame):   has_helmet={has_helmet_full}, confidence={conf_full:.4f}")
    
    # Strategy 2: Upper 60%
    height = frame.shape[0]
    upper_frame = frame[0:int(height*0.6), :]
    upper_resized = cv2.resize(upper_frame, (128, 128))
    has_helmet_upper, conf_upper = detector.detect_helmet(upper_resized)
    print(f"Strategy 2 (Upper 60%):     has_helmet={has_helmet_upper}, confidence={conf_upper:.4f}")
    
    # Strategy 3: Center region
    h, w = frame.shape[:2]
    center_y1, center_y2 = int(h*0.1), int(h*0.7)
    center_x1, center_x2 = int(w*0.2), int(w*0.8)
    center_frame = frame[center_y1:center_y2, center_x1:center_x2]
    center_resized = cv2.resize(center_frame, (128, 128))
    has_helmet_center, conf_center = detector.detect_helmet(center_resized)
    print(f"Strategy 3 (Center Region): has_helmet={has_helmet_center}, confidence={conf_center:.4f}")
    
    print("-" * 60)
    
    # Final decision
    max_confidence = max(conf_full, conf_upper, conf_center)
    helmet_detected = max_confidence > 0.35
    
    print(f"\nMaximum Confidence: {max_confidence:.4f}")
    print(f"Threshold: 0.35")
    print(f"Final Decision: {'HELMET DETECTED ✓' if helmet_detected else 'NO HELMET ✗'}")
    
    # Save frame with results (instead of showing, since cv2.imshow may not work)
    print("\n[Display] Saving result image...")
    
    # Draw results on frame
    cv2.putText(frame, f"Helmet: {helmet_detected}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if helmet_detected else (0, 0, 255), 2)
    cv2.putText(frame, f"Confidence: {max_confidence:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Full: {conf_full:.2f}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Upper: {conf_upper:.2f}", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Center: {conf_center:.2f}", (10, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save image instead of showing
    output_file = 'helmet_test_result.jpg'
    cv2.imwrite(output_file, frame)
    print(f"✓ Result saved to: {output_file}")
    print("  Open this file to see the detection result with confidence values")
    
    # Cleanup
    cap.release()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    
    return helmet_detected

if __name__ == '__main__':
    try:
        result = test_helmet_detection()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
