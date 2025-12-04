# ATM Security System - Complete Application Structure

## File: app_final_complete.py

This is the **complete, fully integrated ATM Security System** with comprehensive comments and section headings for every component.

---

## 📋 TABLE OF CONTENTS

### SECTION 1: Import Statements
- Standard library imports (json, logging, sys, time, signal)
- Third-party imports (cv2, Flask)
- Logging configuration imports

### SECTION 2: AI/ML Detection Models - Computer Vision Modules

#### 2.1 Face Detection Module (Haar Cascade Classifier)
- **Technology**: Viola-Jones algorithm with Haar Cascade
- **Purpose**: Detect human faces in video frames
- **Performance**: ~30ms per frame
- **Model**: haarcascade_frontalface_default.xml

#### 2.2 Face Mask Detection Module (PyTorch CNN)
- **Architecture**: 3 Conv blocks (32→64→128 filters) + FC layers
- **Classes**: with_mask (0) vs without_mask (1)
- **Dataset**: 7,553 images (6,042 train + 1,511 validation)
- **Accuracy**: 97.55% validation
- **Threshold**: 0.5 confidence
- **Model**: models/mask_detector.pth

#### 2.3 Helmet Detection Module (PyTorch CNN)
- **Architecture**: 3 Conv blocks (32→64→128 filters) + FC layers
- **Classes**: with_helmet (0) vs without_helmet (1)
- **Dataset**: 3,925 images (3,140 train + 785 validation)
- **Accuracy**: 99.24% validation
- **Threshold**: 0.4 confidence
- **Model**: models/helmet_detector.pth

#### 2.4 Person Count Detection Module (PyTorch CNN)
- **Architecture**: 3 Conv blocks (32→64→128 filters) + FC layers
- **Classes**: multiple_person (0) vs single_person (1)
- **Dataset**: 40 images (32 train + 8 validation)
- **Accuracy**: 87.50% validation
- **Threshold**: 0.3 confidence
- **Model**: models/person_detector.pth

### SECTION 3: Security & Access Control Modules

#### 3.1 Security Status Manager
- **Purpose**: Centralized security decision-making
- **Logic**: Evaluates all detections and makes access decisions
- **Violations**: no_face, multiple_people, mask_detected, helmet_detected, mask_and_helmet
- **Output**: access_granted (bool) + violation_type (string)

#### 3.2 Audio Alert System
- **Technology**: pygame.mixer for MP3 playback
- **Architecture**: Background thread with queue processing
- **Rate Limiting**: 8 seconds between violation alerts
- **Audio Files**: 
  - multiple_people.mp3
  - mask_detected.mp3
  - helmet_detected.mp3
  - mask_and_helmet.mp3
  - welcome.mp3 (optional)
  - transaction_complete.mp3 (optional)

#### 3.3 ATM State Manager
- **Purpose**: Manage transaction flow and screens
- **Screens**: welcome → pin_entry → menu → withdrawal → processing → complete
- **Business Rules**: 
  - Initial Balance: $5,000
  - Max Withdrawal: $1,000
  - Default PIN: 1234

#### 3.4 Transaction Controller
- **Purpose**: Coordinate security with ATM operations
- **Validation**: Security checked at every transaction step
- **Flow**: Card insertion → PIN entry → Withdrawal → Completion

### SECTION 4: Video Processing & Streaming Module

#### 4.1 Video Stream Handler
- **Purpose**: Generate MJPEG stream with security overlays
- **Pipeline**: 
  1. Frame Capture (640x480 @ 30fps)
  2. Face Detection (Haar Cascade)
  3. ROI Extraction (128x128 for CNNs)
  4. Preprocessing (CLAHE enhancement)
  5. Mask Detection (PyTorch CNN)
  6. Helmet Detection (PyTorch CNN)
  7. Person Detection (PyTorch CNN)
  8. Security Evaluation
  9. Visual Overlays (boxes, text, status)
  10. Audio Alert Queuing
  11. MJPEG Encoding
  12. Memory Cleanup
- **Overlays**: Green/Red boxes, status text, confidence scores

### SECTION 5: Logging System Configuration
- **Console Handler**: INFO and above → stdout
- **File Handler**: ALL levels → logs/atm_system.log (10MB rotation)
- **Error Handler**: ERROR and above → logs/atm_errors.log (10MB rotation)

### SECTION 6: Camera Initialization & Management
- **initialize_camera()**: Initialize with retry logic (3 attempts)
- **reconnect_camera()**: Reconnect after disconnection
- **Configuration**: 640x480 @ 30fps

### SECTION 7: Configuration Management
- **load_config()**: Load config.json
- **Parameters**: Flask port, camera settings, detection thresholds, audio config, ATM rules

### SECTION 8: Flask Application Initialization
- **create_app()**: Initialize Flask app and all modules
- **Sequence**: Config → Camera → Detectors → Security → Audio → ATM → Video → Routes

### SECTION 9: Flask Routes - Web Interface & API

#### 9.1 Web Interface Route
- **GET /**: Serve main ATM interface HTML

#### 9.2 Video Streaming Route
- **GET /video**: MJPEG video stream with overlays

#### 9.3 API Endpoints
- **GET /api/status**: Get ATM and security status
- **POST /api/insert_card**: Handle card insertion
- **POST /api/enter_pin**: Validate PIN entry
- **POST /api/select_withdrawal**: Navigate to withdrawal
- **POST /api/withdraw**: Process withdrawal transaction
- **POST /api/reset**: Reset ATM to welcome screen

### SECTION 10: Resource Cleanup & Shutdown
- **cleanup_resources()**: Graceful shutdown of all modules
- **Sequence**: Audio → Camera → OpenCV windows

### SECTION 11: Main Entry Point
- **main()**: Application startup and server launch
- **Signal Handling**: SIGINT, SIGTERM for graceful shutdown
- **Server**: Flask on 0.0.0.0:5004

### SECTION 12: Script Execution
- **if __name__ == '__main__'**: Entry point when run directly

---

## 🎯 KEY FEATURES

### Security Logic
- **GRANT ACCESS**: Single person, clear face (no mask, no helmet)
- **DENY ACCESS**: Multiple people, masked face, helmeted person, no face

### Detection Models
1. **Face Detection**: Haar Cascade (Viola-Jones)
2. **Mask Detection**: PyTorch CNN (97.55% accuracy)
3. **Helmet Detection**: PyTorch CNN (99.24% accuracy)
4. **Person Count**: PyTorch CNN (87.50% accuracy)

### Transaction Flow
1. Insert Card → Security Check
2. Enter PIN → Security Check
3. Select Withdrawal → Security Check
4. Process Transaction → Security Check
5. Complete → Audio Alert → Reset

### Audio Alerts
- **Violations**: Rate-limited (8 seconds)
- **System Messages**: Immediate playback
- **Background Processing**: Non-blocking

---

## 🚀 USAGE

### Run the Application
```bash
python app_final_complete.py
```

### Access the Interface
```
http://localhost:5004
```

### Stop the Application
```
Press Ctrl+C
```

---

## 📁 REQUIRED FILES

### Models (models/ directory)
- mask_detector.pth
- helmet_detector.pth
- person_detector.pth
- haarcascade_frontalface_default.xml

### Audio (audio/ directory)
- multiple_people.mp3
- mask_detected.mp3
- helmet_detected.mp3
- mask_and_helmet.mp3
- welcome.mp3 (optional)
- transaction_complete.mp3 (optional)

### Configuration
- config.json

### Templates
- templates/index.html

---

## 📊 SYSTEM ARCHITECTURE

```
User Interface (Browser)
    ↓
Flask Web Server (Port 5004)
    ↓
Video Stream Handler ← Camera
    ↓
Detection Pipeline:
    1. Face Detector (Haar Cascade)
    2. Mask Detector (PyTorch CNN)
    3. Helmet Detector (PyTorch CNN)
    4. Person Detector (PyTorch CNN)
    ↓
Security Status Manager
    ↓
Transaction Controller ← ATM State Manager
    ↓
Audio Alert System (Background Thread)
```

---

## ✅ COMPLETE INTEGRATION

This single file contains:
- ✅ All imports and dependencies
- ✅ All AI/ML model integrations
- ✅ Complete security logic
- ✅ Full ATM transaction flow
- ✅ Video processing pipeline
- ✅ Audio alert system
- ✅ Flask web server and API
- ✅ Error handling and recovery
- ✅ Resource cleanup
- ✅ Comprehensive comments

**Total Lines**: ~800+ lines of fully commented, production-ready code!
