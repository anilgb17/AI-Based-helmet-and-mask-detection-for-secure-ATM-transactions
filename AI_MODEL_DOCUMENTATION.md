# ATM Security System - AI Model Documentation

## 1. Which Model is Used for Training?

### Answer: Custom CNN (Convolutional Neural Network)

We built a **Custom CNN from scratch using PyTorch** for training the data.

**NOT using:**
- ❌ YOLO (used for object detection with bounding boxes)
- ❌ ResNet (pre-trained model)
- ❌ VGG (pre-trained model)
- ❌ MobileNet (pre-trained model)

**We ARE using:**
- ✅ Custom CNN (built from scratch for image classification)
- ✅ Haar Cascade (for face detection only)

---

## 2. Model Architecture Details

### Custom CNN Architecture

```
Input: 128×128×3 RGB Image
    ↓
[Convolutional Block 1]
- Conv2D (3 → 32 filters)
- ReLU Activation
- Batch Normalization
- MaxPooling (2×2)
- Dropout (0.25)
    ↓
[Convolutional Block 2]
- Conv2D (32 → 64 filters)
- ReLU Activation
- Batch Normalization
- MaxPooling (2×2)
- Dropout (0.25)
    ↓
[Convolutional Block 3]
- Conv2D (64 → 128 filters)
- ReLU Activation
- Batch Normalization
- MaxPooling (2×2)
- Dropout (0.25)
    ↓
[Fully Connected Layers]
- Flatten
- Dense (32,768 → 128)
- ReLU + Batch Normalization
- Dropout (0.5)
- Dense (128 → 2)
    ↓
Output: 2 Classes (Binary Classification)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch 2.9.1 |
| Image Size | 128×128 pixels |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross-Entropy Loss |
| Scheduler | ReduceLROnPlateau |

---

## 3. Three Trained Models

### Model 1: Mask Detection
- **File**: `models/mask_detector.pth`
- **Dataset**: 7,553 images (6,042 train / 1,511 validation)
- **Classes**: `with_mask`, `without_mask`
- **Accuracy**: 97.22%
- **Training Script**: `train_mask_model.py`

### Model 2: Helmet Detection
- **File**: `models/helmet_detector.pth`
- **Dataset**: 3,925 images (3,140 train / 785 validation)
- **Classes**: `with_helmet`, `without_helmet`
- **Accuracy**: 99.11%
- **Training Script**: `train_helmet_model.py`

### Model 3: Person Detection
- **File**: `models/person_detector.pth`
- **Dataset**: 40 images (32 train / 8 validation)
- **Classes**: `single_person`, `multiple_person`
- **Accuracy**: 50% (needs more data)
- **Training Script**: `train_person_model.py`

---

## 4. Where is AI Used in This Project?

### YES! AI is Used Extensively

The project uses **4 AI/ML models** for security detection:

### AI Component #1: Face Detection
- **Location**: `detection/face_detector.py`
- **Model**: Haar Cascade Classifier (Machine Learning)
- **Purpose**: Detect human faces in camera frames
- **Type**: Computer Vision / Object Detection

### AI Component #2: Mask Detection
- **Location**: `detection/mask_detector_pytorch.py`
- **Model**: Custom CNN (Deep Learning)
- **Purpose**: Classify if person is wearing a mask
- **Type**: Image Classification
- **Accuracy**: 97.22%

### AI Component #3: Helmet Detection
- **Location**: `detection/helmet_detector_pytorch.py`
- **Model**: Custom CNN (Deep Learning)
- **Purpose**: Classify if person is wearing a helmet
- **Type**: Image Classification
- **Accuracy**: 99.11%

### AI Component #4: Person Detection
- **Location**: `detection/person_detector_pytorch.py`
- **Model**: Custom CNN (Deep Learning)
- **Purpose**: Detect single vs multiple persons
- **Type**: Image Classification

---

## 5. How AI Works in the System

### System Flow with AI

```
Step 1: Camera captures video frame
           ↓
Step 2: [AI] Haar Cascade detects faces
           ↓
Step 3: [AI] Mask CNN checks for mask (97.22% accuracy)
           ↓
Step 4: [AI] Helmet CNN checks for helmet (99.11% accuracy)
           ↓
Step 5: [AI] Person CNN checks for multiple people
           ↓
Step 6: Security decision → Allow or Block ATM access
```

### AI Decision Logic

The AI blocks ATM access when:
- ❌ No face detected
- ❌ Multiple people detected
- ❌ Mask detected on face
- ❌ Helmet detected on face
- ❌ Both mask and helmet detected

---

## 6. Why Custom CNN Instead of YOLO?

| Feature | YOLO | Custom CNN | Our Choice |
|---------|------|------------|------------|
| Purpose | Object Detection (bounding boxes) | Image Classification | ✅ Classification |
| Task | Find objects in image | Classify entire image | ✅ Classify faces |
| Speed | Fast but complex | Very fast and simple | ✅ Real-time |
| Model Size | Large (100+ MB) | Small (17 MB) | ✅ Lightweight |
| Training | Needs bbox annotations | Needs class labels | ✅ Simpler data |

**We chose Custom CNN because:**
1. Our task is **classification**, not detection
2. We already use Haar Cascade for face detection
3. Smaller model = faster real-time inference
4. Sufficient accuracy for binary classification
5. Easier to train with labeled images

---

## 7. Training the Models

### Train All Models at Once
```bash
python train_all_models.py
```

### Train Individual Models
```bash
python train_mask_model.py
python train_helmet_model.py
python train_person_model.py
```

### Training Results
- Mask Model: 97.22% validation accuracy ✅
- Helmet Model: 99.11% validation accuracy ✅
- Person Model: 50% validation accuracy ⚠️ (needs more data)

---

## 8. Files Where AI is Implemented

### Training Scripts
- `train_mask_model.py` - Train mask detection CNN
- `train_helmet_model.py` - Train helmet detection CNN
- `train_person_model.py` - Train person detection CNN
- `train_all_models.py` - Train all models sequentially

### Detection Modules (AI Inference)
- `detection/face_detector.py` - Haar Cascade face detection
- `detection/mask_detector_pytorch.py` - Mask classification
- `detection/helmet_detector_pytorch.py` - Helmet classification
- `detection/person_detector_pytorch.py` - Person detection

### Trained Models
- `models/mask_detector.pth` - Trained mask CNN weights
- `models/helmet_detector.pth` - Trained helmet CNN weights
- `models/person_detector.pth` - Trained person CNN weights

---

## 9. Quick Answer for Your Lecturer

### Question 1: Which model did you use for training?
**Answer**: "We used a Custom Convolutional Neural Network (CNN) built from scratch using PyTorch. The architecture has 3 convolutional blocks with 32, 64, and 128 filters, trained for 20 epochs using Adam optimizer."

### Question 2: Is AI used in this project?
**Answer**: "Yes, AI is the core of this project. We use 4 AI models: Haar Cascade for face detection, and 3 custom CNNs for mask detection (97.22% accuracy), helmet detection (99.11% accuracy), and person detection. These AI models work together to make intelligent security decisions in real-time."

### Question 3: Why not use YOLO or ResNet?
**Answer**: "YOLO is for object detection with bounding boxes, but our task is image classification. We already use Haar Cascade for face detection, then our custom CNN classifies the detected faces. Custom CNN is smaller, faster, and sufficient for binary classification tasks."

---

## 10. Summary

✅ **Model Used**: Custom CNN (Convolutional Neural Network)
✅ **Framework**: PyTorch 2.9.1
✅ **AI Components**: 4 models (1 Haar Cascade + 3 CNNs)
✅ **Purpose**: Real-time security classification
✅ **Accuracy**: 97-99% for mask and helmet detection
✅ **Deployment**: Real-time inference in ATM security system

**The entire security system is powered by AI/Deep Learning!** 🤖
