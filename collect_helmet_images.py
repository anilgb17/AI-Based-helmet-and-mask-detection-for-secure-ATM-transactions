"""
Collect Helmet Images for Training
Captures images from camera and saves them for model retraining
"""
import cv2
import os
from datetime import datetime

# Configuration
OUTPUT_DIR = "helmet dataset/data/with_helmet/motorcycle"
NUM_IMAGES = 200
CAPTURE_INTERVAL = 1  # seconds

def collect_images():
    """Collect helmet images from camera"""
    print("=" * 70)
    print("HELMET IMAGE COLLECTION")
    print("=" * 70)
    print(f"\nWill capture {NUM_IMAGES} images")
    print(f"Images will be saved to: {OUTPUT_DIR}")
    print("\nInstructions:")
    print("1. Wear your motorcycle helmet")
    print("2. Move your head slowly (left, right, up, down)")
    print("3. Try different angles and distances")
    print("4. Press 'q' to quit early")
    print("\nPress ENTER to start...")
    input()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("\nCapturing images...")
    print("Move your head to different positions")
    
    count = 0
    last_capture_time = 0
    
    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break
        
        # Display frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Images: {count}/{NUM_IMAGES}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Move your head slowly", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Try to show frame (may not work on headless systems)
        try:
            cv2.imshow('Collecting Helmet Images', display_frame)
        except:
            pass
        
        # Capture image at intervals
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"motorcycle_helmet_{timestamp}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            
            count += 1
            last_capture_time = current_time
            print(f"Captured {count}/{NUM_IMAGES}: {filename}")
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCapture stopped by user")
            break
    
    # Cleanup
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    print("\n" + "=" * 70)
    print(f"COLLECTION COMPLETE: {count} images saved")
    print("=" * 70)
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review images and delete bad ones")
    print("2. Run: python train_helmet_model.py")
    print("3. Test new model with: python test_helmet_simple.py")

if __name__ == '__main__':
    try:
        collect_images()
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
