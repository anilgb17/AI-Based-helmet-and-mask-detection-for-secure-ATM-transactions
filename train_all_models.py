"""
Train both mask and helmet detection models
"""
import sys
import time

def train_all():
    """Train all models sequentially"""
    print("\n" + "=" * 70)
    print(" TRAINING MASK, HELMET, AND PERSON DETECTION MODELS")
    print("=" * 70 + "\n")
    
    # Train mask detection model
    print("\n[1/3] Training Mask Detection Model...")
    print("-" * 70)
    try:
        import train_mask_model
        train_mask_model.train_model()
        print("\n✓ Mask detection model training completed!")
    except Exception as e:
        print(f"\n✗ Error training mask model: {e}")
        return False
    
    time.sleep(2)
    
    # Train helmet detection model
    print("\n\n[2/3] Training Helmet Detection Model...")
    print("-" * 70)
    try:
        import train_helmet_model
        train_helmet_model.train_model()
        print("\n✓ Helmet detection model training completed!")
    except Exception as e:
        print(f"\n✗ Error training helmet model: {e}")
        return False
    
    time.sleep(2)
    
    # Train person detection model
    print("\n\n[3/3] Training Person Detection Model...")
    print("-" * 70)
    try:
        import train_person_model
        train_person_model.train_model()
        print("\n✓ Person detection model training completed!")
    except Exception as e:
        print(f"\n✗ Error training person model: {e}")
        return False
    
    print("\n" + "=" * 70)
    print(" ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 70)
    print("\nTrained models saved in 'models/' directory:")
    print("  - models/mask_detector.pth")
    print("  - models/helmet_detector.pth")
    print("  - models/person_detector.pth")
    print("\nTraining plots saved:")
    print("  - models/mask_training_plot.png")
    print("  - models/helmet_training_plot.png")
    print("  - models/person_training_plot.png")
    print("=" * 70 + "\n")
    
    return True

if __name__ == '__main__':
    success = train_all()
    sys.exit(0 if success else 1)
