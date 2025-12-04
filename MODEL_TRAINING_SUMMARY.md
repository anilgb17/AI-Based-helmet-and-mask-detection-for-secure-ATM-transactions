# ATM Security System - Model Training Summary

## Overview
Successfully trained and integrated three PyTorch CNN models for the ATM Security System.

## Training Results

### 1. Mask Detection Model
- **Dataset**: `mask dataset/data`
  - Classes: `with_mask`, `without_mask`
  - Training samples: 6,042
  - Validation samples: 1,511
- **Performance**: 97.55% validation accuracy
- **Model**: `models/mask_detector.pth`
- **Training Plot**: `models/mask_training_plot.png`

### 2. Helmet Detection Model
- **Dataset**: `helmet dataset/data`
  - Classes: `with_helmet`, `without_helmet`
  - Training samples: 3,140
  - Validation samples: 785
- **Performance**: 99.24% validation accuracy
- **Model**: `models/helmet_detector.pth`
- **Training Plot**: `models/helmet_training_plot.png`

### 3. Person Detection Model
- **Dataset**: `person dataset/data`
  - Classes: `multiple_person`, `single_person`
  - Training samples: 32
  - Validation samples: 8
- **Performance**: 87.50% validation accuracy
- **Model**: `models/person_detector.pth`
- **Training Plot**: `models/person_training_plot.png`

## Model Architecture
All three models use the same CNN architecture:
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch Normalization and Dropout for regularization
- Fully connected layers with 128 hidden units
- Binary classification output (2 classes)

## Training Configuration
- **Image Size**: 128x128 pixels
- **Batch Size**: 32
- **Epochs**: 20
- **Learning Rate**: 0.001 (Adam optimizer)
- **Data Augmentation**: Rotation, flipping, color jitter
- **Framework**: PyTorch 2.9.1

## Integration
The models are integrated into the ATM Security System:
- **Mask Detector**: `detection/mask_detector_pytorch.py`
- **Helmet Detector**: `detection/helmet_detector_pytorch.py`
- **Person Detector**: `detection/person_detector_pytorch.py`

## Security Logic
The system denies access when:
1. No face detected
2. Multiple people detected (face count > 1 OR person detector triggers)
3. Mask detected
4. Helmet detected
5. Both mask and helmet detected

## Running the System

### Train All Models
```bash
python train_all_models.py
```

### Train Individual Models
```bash
python train_mask_model.py
python train_helmet_model.py
python train_person_model.py
```

### Run the Application
```bash
python app.py
```

Access the web interface at:
- Local: http://127.0.0.1:5004
- Network: http://10.183.123.67:5004

## Files Created
- `train_mask_model.py` - Mask detection training script
- `train_helmet_model.py` - Helmet detection training script
- `train_person_model.py` - Person detection training script
- `train_all_models.py` - Train all models sequentially
- `detection/mask_detector_pytorch.py` - PyTorch mask detector
- `detection/helmet_detector_pytorch.py` - PyTorch helmet detector
- `detection/person_detector_pytorch.py` - PyTorch person detector

## System Status
✅ All models trained successfully
✅ All models integrated into the application
✅ Application running with real-time detection
✅ Multi-person detection active
✅ Security monitoring operational
