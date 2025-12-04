"""
Complete System Diagnostic Tool
Tests all detection components and shows what's being detected
"""
import cv2
import sys
from detection.face_detector import FaceDetector
from detection.mask_detector_pytorch import MaskDetectorPyTorch as MaskDetector
from detection.helmet_detector_pytorch import HelmetDetectorPyTorch as HelmetDetector
from detection.person_detector_pytorch import PersonDetectorPyTorch as PersonDetector

def diagnose():
    """Run complete system diagnostic"""
    print("=" * 80)
    print("ATM SECURITY SYSTEM - COMPLETE DIAGNOSTIC")
    print("=" * 80)
    
    # Initialize all detectors
    print("\n[1/5] Initializing detectors...")
    try:
        face_detector = FaceDetector()
        mask_detector = MaskDetector(threshold=0.5)
        helmet_detector = HelmetDetector(threshold=0.4)
        person_detector = PersonDetector(threshold=0.3)
        print("✓ All detectors initialized")
    except Exception as e:
        print(f"✗ Failed to initialize detectors: {e}")
        return False
    
    # Open camera
    print("\n[2/5] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return False
    print("✓ Camera opened")
    
    # Capture frame
    print("\n[3/5] Capturing frame...")
    ret, frame = cap.read()
    if not ret or frame is None:
        print("✗ Failed to capture frame")
        cap.release()
        return False
    print(f"✓ Frame captured: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run all detections
    print("\n[4/5] Running detections...")
    print("-" * 80)
    
    # Face detection
    print("\n[FACE DETECTION]")
    faces = face_detector.detect_faces(frame)
    face_count = len(faces)
    print(f"  Faces detected: {face_count}")
    if face_count > 0:
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  Face {i+1}: position=({x},{y}), size={w}x{h}")
    
    # Person detection
    print("\n[PERSON DETECTION]")
    try:
        is_multiple, person_conf = person_detector.detect_multiple_persons(frame)
        print(f"  Multiple persons: {is_multiple}")
        print(f"  Confidence: {person_conf:.4f}")
        print(f"  Threshold: 0.3")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Mask and helmet detection (if face detected)
    if face_count > 0:
        print("\n[MASK & HELMET DETECTION - Face Region]")
        face_rect = faces[0]
        face_roi = face_detector.extract_face_roi(frame, face_rect)
        
        if face_roi is not None:
            preprocessed_face = face_detector.preprocess_face(face_roi)
            
            if preprocessed_face is not None:
                # Mask detection
                try:
                    has_mask, mask_conf = mask_detector.detect_mask(preprocessed_face)
                    print(f"  Mask detected: {has_mask}")
                    print(f"  Mask confidence: {mask_conf:.4f}")
                    print(f"  Mask threshold: 0.5")
                except Exception as e:
                    print(f"  Mask detection error: {e}")
                
                # Helmet detection
                try:
                    has_helmet, helmet_conf = helmet_detector.detect_helmet(preprocessed_face)
                    print(f"  Helmet detected: {has_helmet}")
                    print(f"  Helmet confidence: {helmet_conf:.4f}")
                    print(f"  Helmet threshold: 0.4")
                except Exception as e:
                    print(f"  Helmet detection error: {e}")
    else:
        print("\n[MASK & HELMET DETECTION - Full Frame Fallback]")
        # Full frame detection
        frame_resized = cv2.resize(frame, (128, 128))
        
        # Mask on full frame
        try:
            has_mask_full, mask_conf_full = mask_detector.detect_mask(frame_resized)
            print(f"  Mask (full frame): {has_mask_full}, confidence={mask_conf_full:.4f}")
        except Exception as e:
            print(f"  Mask detection error: {e}")
        
        # Helmet on full frame
        try:
            has_helmet_full, helmet_conf_full = helmet_detector.detect_helmet(frame_resized)
            print(f"  Helmet (full frame): {has_helmet_full}, confidence={helmet_conf_full:.4f}")
        except Exception as e:
            print(f"  Helmet detection error: {e}")
        
        # Helmet on upper frame
        height = frame.shape[0]
        upper_frame = frame[0:int(height*0.6), :]
        upper_resized = cv2.resize(upper_frame, (128, 128))
        try:
            has_helmet_upper, helmet_conf_upper = helmet_detector.detect_helmet(upper_resized)
            print(f"  Helmet (upper 60%): {has_helmet_upper}, confidence={helmet_conf_upper:.4f}")
        except Exception as e:
            print(f"  Helmet detection error: {e}")
        
        # Helmet on center region
        h, w = frame.shape[:2]
        center_y1, center_y2 = int(h*0.1), int(h*0.7)
        center_x1, center_x2 = int(w*0.2), int(w*0.8)
        center_frame = frame[center_y1:center_y2, center_x1:center_x2]
        center_resized = cv2.resize(center_frame, (128, 128))
        try:
            has_helmet_center, helmet_conf_center = helmet_detector.detect_helmet(center_resized)
            print(f"  Helmet (center region): {has_helmet_center}, confidence={helmet_conf_center:.4f}")
        except Exception as e:
            print(f"  Helmet detection error: {e}")
    
    # Security decision
    print("\n[5/5] Security Decision")
    print("-" * 80)
    
    # Determine violations
    violations = []
    
    if face_count == 0:
        violations.append("NO_FACE_DETECTED")
    elif face_count >= 2:
        violations.append("MULTIPLE_PEOPLE")
    
    if is_multiple and person_conf > 0.3:
        violations.append("MULTIPLE_PEOPLE (CNN)")
    
    if face_count > 0 and has_mask:
        violations.append("MASK_DETECTED")
    
    if face_count > 0 and has_helmet:
        violations.append("HELMET_DETECTED")
    
    if face_count > 0 and has_mask and has_helmet:
        violations.append("MASK_AND_HELMET_DETECTED")
    
    # Final decision
    if len(violations) == 0:
        print("  Decision: ✓ ACCESS GRANTED")
        print("  Reason: Single person, clear face, no violations")
    else:
        print("  Decision: ✗ ACCESS DENIED")
        print(f"  Violations: {', '.join(violations)}")
    
    # Save diagnostic image
    print("\n[SAVING DIAGNOSTIC IMAGE]")
    output_file = 'diagnostic_result.jpg'
    
    # Draw on frame
    for (x, y, w, h) in faces:
        color = (0, 255, 0) if len(violations) == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Add text
    status_text = "ACCESS GRANTED" if len(violations) == 0 else "ACCESS DENIED"
    status_color = (0, 255, 0) if len(violations) == 0 else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if len(violations) > 0:
        y_pos = 110
        for violation in violations:
            cv2.putText(frame, violation, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_pos += 30
    
    cv2.imwrite(output_file, frame)
    print(f"✓ Diagnostic image saved to: {output_file}")
    
    # Cleanup
    cap.release()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    try:
        result = diagnose()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nDiagnostic failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
