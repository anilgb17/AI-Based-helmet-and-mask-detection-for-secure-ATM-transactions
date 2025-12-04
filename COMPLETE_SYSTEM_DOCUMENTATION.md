# ATM Security System - Complete Documentation with All Models

## 📋 Table of Contents

1. [Face Detection Model](#1-face-detection-model)
2. [Mask Detection Model](#2-mask-detection-model)
3. [Helmet Detection Model](#3-helmet-detection-model)
4. [Person Detection Model](#4-person-detection-model)
5. [Video Processing Pipeline](#5-video-processing-pipeline)
6. [Security Violation System](#6-security-violation-system)
7. [ATM Interface](#7-atm-interface)
8. [Audio Alert System](#8-audio-alert-system)
9. [Complete System Flow](#9-complete-system-flow)

---

## 1. FACE DETECTION MODEL

### Technology
- **Algorithm**: Haar Cascade Classifier (Viola-Jones)
- **Type**: Traditional Computer Vision (not deep learning)
- **File**: `detection/face_detector.py`

### Purpose
Detect human faces in video frames for security monitoring

### Parameters (Optimized for Multiple Person Detection)
```python
scaleFactor = 1.15    # Balanced sensitivity
minNeighbors = 5      # Moderate evidence required
minSize = (40, 40)    # Detects smaller/distant faces
max_faces = 3         # Keep up to 3 largest faces
```

### Input/Output
- **Input**: BGR video frame (640x480)
- **Output**: List of face rectangles `[(x, y, w, h), ...]`

### Processing Steps
1. Convert frame to grayscale
2. Apply Haar Cascade detection
3. Filter overlapping faces (>30% overlap)
4. Keep 3 largest faces
5. Return face list

### Performance
- **Speed**: <50ms per frame
- **Accuracy**: High for frontal faces, moderate for profiles

---

## 2. MASK DETECTION MODEL

### Model Architecture
```
INPUT: 128x128x3 RGB Image
    ↓
Conv2D(32) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Linear(128) → ReLU → BatchNorm → Dropout(0.5)
    ↓
Linear(2) → Softmax
    ↓
OUTPUT: 2 classes
```

### Classes
- **Class 0**: `with_mask` - Person wearing face mask
- **Class 1**: `without_mask` - Person without face mask

### Training Details
- **Dataset**: 7,553 images
  - Training: 6,042 images
  - Validation: 1,511 images
- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Validation Accuracy**: **97.55%**

### Data Augmentation
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random affine transformations

### Input/Output
- **Input**: 128x128 RGB face image (preprocessed with CLAHE)
- **Output**: `(has_mask: bool, confidence: float)`
- **Threshold**: 0.5 (configurable)

### Model File
- **Path**: `models/mask_detector.pth`
- **Size**: ~2MB
- **Framework**: PyTorch 2.0+

### Usage in System
```python
has_mask, confidence = mask_detector.detect_mask(face_roi)
if has_mask and confidence > 0.5:
    # Deny access - mask detected
```

---

## 3. HELMET DETECTION MODEL

### Model Architecture
```
INPUT: 128x128x3 RGB Image
    ↓
Conv2D(32) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Linear(128) → ReLU → BatchNorm → Dropout(0.5)
    ↓
Linear(2) → Softmax
    ↓
OUTPUT: 2 classes
```

### Classes
- **Class 0**: `with_helmet` - Person wearing helmet/head covering
- **Class 1**: `without_helmet` - Person without helmet

### Training Details
- **Dataset**: 3,925 images
  - Training: 3,140 images
  - Validation: 785 images
- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Validation Accuracy**: **99.24%**

### Data Augmentation
- Random horizontal flip
- Random rotation (±15°)
- Color jitter
- Random affine transformations

### Input/Output
- **Input**: 128x128 RGB face image (preprocessed with CLAHE)
- **Output**: `(has_helmet: bool, confidence: float)`
- **Threshold**: 0.4 (configurable)

### Model File
- **Path**: `models/helmet_detector.pth`
- **Size**: ~2MB
- **Framework**: PyTorch 2.0+

### Usage in System
```