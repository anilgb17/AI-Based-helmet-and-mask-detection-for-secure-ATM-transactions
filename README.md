# ATM Security System

A real-time facial recognition and security monitoring system designed to prevent fraudulent ATM transactions by detecting and blocking users wearing masks, helmets, or other facial coverings.

## Overview

The ATM Security System combines computer vision, real-time video processing, and web technologies to create a secure ATM interface. The system uses OpenCV for facial detection and analysis, Flask for the web framework, and pygame for audio alerts.

## Features

- **Real-time Face Detection**: Continuous monitoring using Haar Cascade classifiers
- **Mask Detection**: Multi-method analysis to detect face masks
- **Helmet Detection**: Detection of helmets and head coverings
- **Audio Alerts**: Voice warnings for security violations
- **Web-based ATM Interface**: Complete transaction flow with PIN validation
- **Video Streaming**: Live camera feed with security overlays
- **Access Control**: Automatic transaction blocking on security violations

## System Requirements

### Hardware
- **CPU**: Dual-core processor, 2.0 GHz or higher
- **RAM**: Minimum 2GB, recommended 4GB
- **Camera**: USB webcam or built-in camera, minimum 720p resolution
- **Storage**: 500MB for application and dependencies
- **Audio**: System audio output for alerts

### Software
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 10.15+
- **Python**: 3.8 or higher
- **pip**: Python package manager

## Installation

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd atm-security-system
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- OpenCV (computer vision)
- NumPy (numerical operations)
- pygame (audio playback)
- psutil (system monitoring)

### 3. Set Up Audio Files

The system requires 4 MP3 audio files for security alerts. Place these files in the `audio/` directory:

- `multiple_people.mp3` - Warning for multiple people detected
- `mask_detected.mp3` - Warning for face mask detected
- `helmet_detected.mp3` - Warning for helmet detected
- `mask_and_helmet.mp3` - Warning for both mask and helmet detected

See `audio/README.md` for detailed audio file specifications and guidelines.

### 4. Verify Camera Access

Ensure your camera is connected and accessible. The system uses camera index 0 by default (usually the built-in or first USB camera).

Test camera access:
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### 5. Configure the System (Optional)

Edit `config.json` to customize system settings. See the Configuration section below for details.

## Running the Application

### Start the Server

```bash
python app.py
```

The application will start on `http://localhost:5004` by default.

### Access the ATM Interface

Open a web browser and navigate to:
```
http://localhost:5004
```

You should see:
- The ATM interface with a welcome screen
- Live video feed showing camera output
- Security status indicators

### Default Credentials

- **PIN**: 1234
- **Initial Balance**: $5000.00
- **Max Withdrawal**: $1000.00

## Configuration

The system is configured through `config.json`. Here are the available options:

### Camera Configuration

```json
"camera": {
  "index": 0,           // Camera device index (0 = default camera)
  "width": 640,         // Frame width in pixels
  "height": 480,        // Frame height in pixels
  "fps": 30             // Target frames per second
}
```

### Detection Thresholds

```json
"detection": {
  "mask_threshold": 0.5,      // Confidence threshold for mask detection (0.0-1.0)
  "helmet_threshold": 0.4,    // Confidence threshold for helmet detection (0.0-1.0)
  "face_cascade": "haarcascade_frontalface_default.xml"  // Haar Cascade file
}
```

**Adjusting Thresholds**:
- Lower values = more sensitive (more detections, possible false positives)
- Higher values = less sensitive (fewer detections, possible false negatives)
- Default values (0.5 for mask, 0.4 for helmet) are recommended

### Audio Configuration

```json
"audio": {
  "enabled": true,              // Enable/disable audio alerts
  "rate_limit_seconds": 8,      // Minimum seconds between alerts
  "audio_directory": "audio"    // Path to audio files directory
}
```

### ATM Configuration

```json
"atm": {
  "initial_balance": 5000.00,   // Starting account balance
  "max_withdrawal": 1000.00,    // Maximum withdrawal per transaction
  "correct_pin": "1234"         // Valid PIN (change for production!)
}
```

### Flask Configuration

```json
"flask_port": 5004              // Web server port
```

## Usage

### Normal Transaction Flow

1. **Insert Card**: Click "Insert Card" button
   - System verifies no security violations
   - Proceeds to PIN entry if face is clear

2. **Enter PIN**: Use on-screen keypad to enter PIN (default: 1234)
   - System continuously monitors security status
   - Transaction cancelled if violation detected

3. **Select Transaction**: Choose from menu options
   - Withdrawal
   - Balance Inquiry
   - Statement
   - Exit

4. **Process Withdrawal**: Enter amount and confirm
   - Amount must be > $0
   - Amount must be ≤ current balance
   - Amount must be ≤ $1000
   - Security status verified before processing

5. **Complete**: View confirmation and new balance
   - System automatically returns to welcome screen

### Security Violations

The system will block transactions if:
- **No face detected**: No person visible in camera
- **Multiple people**: More than one person detected
- **Mask detected**: Face mask covering mouth/nose
- **Helmet detected**: Helmet or head covering detected
- **Both mask and helmet**: Multiple coverings detected

When a violation occurs:
- Audio alert plays (if enabled)
- Visual warning displayed on screen
- Transaction is blocked or cancelled
- User must resolve violation to proceed

## API Endpoints

The system provides REST API endpoints for integration:

### GET /
Serves the main ATM interface HTML page

### GET /video
Streams live video feed with security overlays (MJPEG format)

### GET /api/status
Returns current ATM and security status
```json
{
  "screen": "welcome",
  "balance": 5000.00,
  "security": {
    "access_granted": true,
    "violation_type": null,
    "face_count": 1,
    "has_mask": false,
    "has_helmet": false,
    "mask_confidence": 0.12,
    "helmet_confidence": 0.08
  }
}
```

### POST /api/insert_card
Attempts to insert card and proceed to PIN entry

### POST /api/enter_pin
Validates PIN and proceeds to main menu
```json
{"pin": "1234"}
```

### POST /api/select_withdrawal
Navigates to withdrawal screen

### POST /api/withdraw
Processes withdrawal transaction
```json
{"amount": 100.00}
```

### POST /api/reset
Resets ATM to welcome screen

## Troubleshooting

### Camera Not Working
- Verify camera is connected and not in use by another application
- Try different camera index in `config.json` (0, 1, 2, etc.)
- Check camera permissions on your operating system
- Test camera with: `python -c "import cv2; cv2.VideoCapture(0).read()"`

### Audio Not Playing
- Verify all 4 MP3 files exist in `audio/` directory
- Check file names match exactly (case-sensitive)
- Ensure pygame is properly installed: `pip install pygame`
- Test system audio output with other applications
- Check `audio.enabled` setting in `config.json`

### Detection Not Working
- Ensure adequate lighting for camera
- Position face directly in front of camera
- Verify Haar Cascade file exists (included with OpenCV)
- Check detection thresholds in `config.json`
- Review console output for error messages

### High CPU Usage
- Reduce camera resolution in `config.json`
- Lower FPS setting
- Close other resource-intensive applications
- Ensure system meets minimum requirements

### Port Already in Use
- Change `flask_port` in `config.json` to different port
- Stop other applications using port 5004
- Use: `python app.py` (Flask will suggest alternative port)

## Project Structure

```
atm-security-system/
├── app.py                          # Main application entry point
├── config.json                     # System configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── detection/                      # Computer vision modules
│   ├── face_detector.py           # Face detection using Haar Cascade
│   ├── mask_detector.py           # Mask detection analysis
│   ├── helmet_detector.py         # Helmet detection analysis
│   └── __init__.py
│
├── security/                       # Security management
│   ├── security_status_manager.py # Access control logic
│   └── __init__.py
│
├── atm/                           # ATM transaction logic
│   ├── atm_state_manager.py      # State and screen management
│   ├── transaction_controller.py  # Transaction coordination
│   └── __init__.py
│
├── audio/                         # Audio alert system
│   ├── audio_alert_system.py     # Audio playback and queueing
│   ├── README.md                  # Audio file documentation
│   ├── multiple_people.mp3        # Audio files (4 required)
│   ├── mask_detected.mp3
│   ├── helmet_detected.mp3
│   └── mask_and_helmet.mp3
│
├── web/                           # Web streaming
│   ├── video_stream_handler.py   # MJPEG video streaming
│   └── __init__.py
│
├── templates/                     # HTML templates
│   └── index.html                # ATM interface
│
├── static/                        # Static web assets
│   ├── style.css                 # Interface styling
│   └── script.js                 # Client-side logic
│
└── tests/                         # Test suite
    ├── test_face_detector.py
    ├── test_mask_detector.py
    ├── test_helmet_detector.py
    ├── test_security_status_manager.py
    ├── test_audio_alert_system.py
    ├── test_atm_state_manager.py
    ├── test_transaction_controller.py
    ├── test_video_stream_handler.py
    ├── test_api_integration.py
    ├── test_frontend_integration.py
    ├── test_error_handling.py
    └── test_performance.py
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_face_detector.py

# Run with verbose output
python -m pytest -v tests/

# Run with coverage report
python -m pytest --cov=. tests/
```

## Security Considerations

### Development vs Production

This system is designed for demonstration and development purposes. For production deployment:

1. **Change Default PIN**: Update `correct_pin` in `config.json`
2. **Use PIN Hashing**: Implement secure PIN storage with hashing
3. **Enable HTTPS**: Deploy with SSL/TLS certificates
4. **Add Authentication**: Implement proper user authentication
5. **Rate Limiting**: Add API rate limiting to prevent abuse
6. **Audit Logging**: Log all transactions and security events
7. **Camera Privacy**: Ensure video is not recorded or transmitted
8. **Network Security**: Deploy behind firewall with restricted access

### Known Limitations

- Face detection may be affected by lighting conditions
- Detection accuracy depends on camera quality
- System can be bypassed with high-quality 3D masks (not in scope)
- PIN is stored in plain text (development only)
- No encryption for data transmission (use HTTPS in production)

## Performance

Expected performance on recommended hardware:
- **Frame Processing**: < 100ms per frame
- **API Response Time**: < 100ms for status endpoint
- **CPU Usage**: < 50% on dual-core processor
- **Memory Usage**: < 500MB
- **Continuous Operation**: Tested for 8+ hours

## License

[Specify your license here]

## Support

For issues, questions, or contributions:
- Check the Troubleshooting section above
- Review `audio/README.md` for audio-specific issues
- See `DEPLOYMENT.md` for production deployment guidance
- Check console output for error messages and logs

## Acknowledgments

- OpenCV for computer vision capabilities
- Flask for web framework
- Haar Cascade classifiers for face detection
- pygame for audio playback
