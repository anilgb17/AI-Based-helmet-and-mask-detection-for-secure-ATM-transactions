# ATM Security System - Framework Documentation

## Project Framework Overview

This ATM Security System is built using **multiple frameworks** working together to create a complete AI-powered security solution.

---

## 1. Main Frameworks Used

### Backend Framework: **Flask**
- **Version**: 2.0.0+
- **Purpose**: Web server and API endpoints
- **What it does**: 
  - Serves the ATM web interface
  - Handles HTTP requests (PIN validation, transactions)
  - Streams live video feed
  - Provides REST API for frontend communication

### Deep Learning Framework: **PyTorch**
- **Version**: 2.0.0+
- **Purpose**: AI model training and inference
- **What it does**:
  - Train custom CNN models
  - Load trained models for real-time detection
  - Perform mask, helmet, and person detection
- **Additional**: torchvision 0.15.0+ for image transformations

### Computer Vision Framework: **OpenCV**
- **Version**: 4.5.0+
- **Purpose**: Image processing and face detection
- **What it does**:
  - Capture video from camera
  - Detect faces using Haar Cascade
  - Process and manipulate images
  - Draw overlays on video frames

### Audio Framework: **Pygame**
- **Version**: 2.5.0+ (pygame-ce)
- **Purpose**: Audio alert system
- **What it does**:
  - Play security warning sounds
  - Manage audio queue
  - Handle audio rate limiting

---

## 2. Complete Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Web Framework** | Flask 2.0+ | HTTP server, routing, API |
| **Frontend** | HTML5 + CSS3 + JavaScript | User interface |
| **AI/ML Framework** | PyTorch 2.0+ | Deep learning models |
| **Computer Vision** | OpenCV 4.5+ | Image processing, face detection |
| **Audio System** | Pygame-CE 2.5+ | Sound alerts |
| **Data Processing** | NumPy 1.19+ | Numerical operations |
| **Visualization** | Matplotlib 3.5+ | Training plots |
| **Image Processing** | Pillow 9.0+ | Image transformations |
| **System Monitoring** | psutil 5.8+ | Performance monitoring |
| **ML Utilities** | scikit-learn 1.0+ | Data preprocessing |

---

## 3. Project Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ATM SECURITY SYSTEM                       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
   │  Flask  │         │ PyTorch │        │ OpenCV  │
   │  Web    │         │   AI    │        │ Vision  │
   │ Server  │         │ Models  │        │ System  │
   └────┬────┘         └────┬────┘        └────┬────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
   │ Camera  │         │  Audio  │        │   ATM   │
   │ Stream  │         │ Alerts  │        │  Logic  │
   └─────────┘         └─────────┘        └─────────┘
```

---

## 4. Framework Responsibilities

### Flask Framework (Backend)
```python
# File: app_final_complete.py
from flask import Flask, Response, jsonify, request, render_template

app = Flask(__name__)

# Routes:
@app.route('/')                    # Serve ATM interface
@app.route('/video')               # Stream video feed
@app.route('/api/status')          # Get system status
@app.route('/api/insert_card')     # Insert card action
@app.route('/api/enter_pin')       # PIN validation
@app.route('/api/withdraw')        # Process withdrawal
```

**Flask handles:**
- HTTP routing
- JSON API responses
- Template rendering
- Video streaming (MJPEG)
- Session management

---

### PyTorch Framework (AI/ML)
```python
# Files: train_mask_model.py, train_helmet_model.py, train_person_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Model Definition
class MaskDetectorCNN(nn.Module):
    def __init__(self):
        super(MaskDetectorCNN, self).__init__()
        self.features = nn.Sequential(...)
        self.classifier = nn.Sequential(...)
```

**PyTorch handles:**
- CNN model architecture
- Model training (backpropagation)
- Model inference (predictions)
- GPU acceleration (if available)
- Model saving/loading (.pth files)

---

### OpenCV Framework (Computer Vision)
```python
# File: detection/face_detector.py
import cv2

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Video Capture
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Image Processing
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

**OpenCV handles:**
- Camera access
- Face detection (Haar Cascade)
- Image preprocessing
- Drawing overlays
- Color conversions

---

### Pygame Framework (Audio)
```python
# File: audio/audio_alert_system.py
import pygame

pygame.mixer.init()
sound = pygame.mixer.Sound('audio/mask_detected.mp3')
sound.play()
```

**Pygame handles:**
- Audio file loading
- Sound playback
- Audio queue management
- Volume control

---

## 5. Project Structure by Framework

```
atm-security-system/
│
├── Flask (Web Framework)
│   ├── app_final_complete.py          # Main Flask application
│   ├── templates/
│   │   └── index.html                 # HTML template
│   ├── static/
│   │   ├── style.css                  # CSS styling
│   │   └── script.js                  # JavaScript
│   └── web/
│       └── video_stream_handler.py    # Video streaming
│
├── PyTorch (AI Framework)
│   ├── train_mask_model.py            # Train mask CNN
│   ├── train_helmet_model.py          # Train helmet CNN
│   ├── train_person_model.py          # Train person CNN
│   ├── detection/
│   │   ├── mask_detector_pytorch.py   # Mask inference
│   │   ├── helmet_detector_pytorch.py # Helmet inference
│   │   └── person_detector_pytorch.py # Person inference
│   └── models/
│       ├── mask_detector.pth          # Trained weights
│       ├── helmet_detector.pth        # Trained weights
│       └── person_detector.pth        # Trained weights
│
├── OpenCV (Computer Vision)
│   └── detection/
│       └── face_detector.py           # Face detection
│
├── Pygame (Audio Framework)
│   └── audio/
│       ├── audio_alert_system.py      # Audio manager
│       └── *.mp3                      # Sound files
│
├── Business Logic
│   ├── atm/
│   │   ├── atm_state_manager.py       # ATM states
│   │   └── transaction_controller.py  # Transactions
│   └── security/
│       └── security_status_manager.py # Security logic
│
└── Configuration
    ├── config.json                    # System config
    └── requirements.txt               # Dependencies
```

---

## 6. Framework Integration Flow

### How Frameworks Work Together

```
1. User opens browser
        ↓
2. Flask serves HTML/CSS/JS (Frontend)
        ↓
3. JavaScript requests video stream
        ↓
4. Flask starts video streaming
        ↓
5. OpenCV captures camera frame
        ↓
6. OpenCV detects faces (Haar Cascade)
        ↓
7. PyTorch analyzes face (Mask CNN)
        ↓
8. PyTorch analyzes face (Helmet CNN)
        ↓
9. PyTorch analyzes frame (Person CNN)
        ↓
10. Security logic makes decision
        ↓
11. Pygame plays audio alert (if violation)
        ↓
12. Flask sends status to frontend
        ↓
13. JavaScript updates UI
```

---

## 7. Dependencies (requirements.txt)

```txt
# Web Framework
flask>=2.0.0                    # Web server and API

# AI/ML Framework
torch>=2.0.0                    # Deep learning
torchvision>=0.15.0             # Image transformations

# Computer Vision
opencv-python>=4.5.0            # Image processing

# Audio Framework
pygame-ce>=2.5.0                # Sound playback

# Data Processing
numpy>=1.19.0                   # Numerical operations
pillow>=9.0.0                   # Image handling

# Visualization
matplotlib>=3.5.0               # Training plots

# Utilities
scikit-learn>=1.0.0             # ML utilities
psutil>=5.8.0                   # System monitoring
```

---

## 8. Framework Versions

| Framework | Version | Release Date | Status |
|-----------|---------|--------------|--------|
| Flask | 2.0+ | May 2021 | ✅ Stable |
| PyTorch | 2.0+ | March 2023 | ✅ Stable |
| OpenCV | 4.5+ | January 2021 | ✅ Stable |
| Pygame-CE | 2.5+ | 2024 | ✅ Stable |
| NumPy | 1.19+ | June 2020 | ✅ Stable |

---

## 9. Why These Frameworks?

### Flask vs Django
✅ **Flask** - Lightweight, flexible, perfect for small-medium projects
❌ Django - Too heavy for this project

### PyTorch vs TensorFlow
✅ **PyTorch** - Easier to learn, better for research, dynamic graphs
❌ TensorFlow - More complex, better for production at scale

### OpenCV vs PIL
✅ **OpenCV** - Specialized for computer vision, has Haar Cascade
❌ PIL/Pillow - General image processing, no face detection

### Pygame vs pydub
✅ **Pygame** - Real-time audio, low latency, easy to use
❌ pydub - Better for audio editing, not real-time playback

---

## 10. Installation

### Install All Frameworks
```bash
pip install -r requirements.txt
```

### Install Individual Frameworks
```bash
# Web Framework
pip install flask>=2.0.0

# AI Framework
pip install torch>=2.0.0 torchvision>=0.15.0

# Computer Vision
pip install opencv-python>=4.5.0

# Audio Framework
pip install pygame-ce>=2.5.0

# Utilities
pip install numpy>=1.19.0 matplotlib>=3.5.0 pillow>=9.0.0
pip install scikit-learn>=1.0.0 psutil>=5.8.0
```

---

## 11. Running the Application

### Start the System
```bash
python app_final_complete.py
```

### What Happens:
1. **Flask** starts web server on port 5004
2. **OpenCV** initializes camera
3. **PyTorch** loads trained models
4. **Pygame** initializes audio system
5. System ready at http://localhost:5004

---

## 12. Framework Summary

### Primary Framework: **Flask** (Web Application)
The entire system is a **Flask web application** that integrates:
- PyTorch for AI
- OpenCV for vision
- Pygame for audio

### Architecture Pattern: **MVC-like**
- **Model**: PyTorch CNNs, ATM state
- **View**: HTML templates, video stream
- **Controller**: Flask routes, business logic

### Design Pattern: **Modular Architecture**
Each framework is isolated in its own module:
- `detection/` - OpenCV + PyTorch
- `atm/` - Business logic
- `audio/` - Pygame
- `web/` - Flask streaming

---

## 13. Quick Answer for Your Lecturer

**Question: What is the framework of this project?**

**Answer**: 

"This project uses a **multi-framework architecture** with Flask as the main web framework:

**Primary Framework**: Flask 2.0+ (Python web framework)

**Supporting Frameworks**:
- **PyTorch 2.0+** - Deep learning framework for AI models
- **OpenCV 4.5+** - Computer vision framework for image processing
- **Pygame-CE 2.5+** - Audio framework for sound alerts

**Architecture**: Flask web application with integrated AI/ML capabilities using PyTorch CNNs, real-time video processing with OpenCV, and audio alerts via Pygame.

**Pattern**: Modular MVC-like architecture where Flask handles routing and API, PyTorch handles AI inference, OpenCV handles vision processing, and Pygame handles audio playback."

---

## 14. Framework Comparison

| Aspect | Framework Used | Alternative | Why Our Choice? |
|--------|---------------|-------------|-----------------|
| Web | Flask | Django, FastAPI | Lightweight, flexible |
| AI/ML | PyTorch | TensorFlow, Keras | Easier, dynamic graphs |
| Vision | OpenCV | PIL, scikit-image | Best for face detection |
| Audio | Pygame | pydub, sounddevice | Real-time, low latency |

---

## Summary

✅ **Main Framework**: Flask (Web Application Framework)
✅ **AI Framework**: PyTorch (Deep Learning)
✅ **Vision Framework**: OpenCV (Computer Vision)
✅ **Audio Framework**: Pygame (Sound System)
✅ **Architecture**: Modular, MVC-like, Multi-framework Integration

**This is a Flask-based web application powered by AI/ML frameworks!** 🚀
