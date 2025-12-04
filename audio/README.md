# Audio Files for ATM Security System

This directory contains the audio alert files used by the ATM Security System to provide voice warnings when security violations are detected.

## Required Audio Files

The system requires the following 4 MP3 files to be present in this directory:

### 1. `multiple_people.mp3`
- **Purpose**: Played when multiple people are detected in the camera frame
- **Violation Type**: `multiple_people`
- **Suggested Message**: "Warning: Multiple people detected. Only one person is allowed at the ATM."

### 2. `mask_detected.mp3`
- **Purpose**: Played when a face mask is detected on the user
- **Violation Type**: `mask_detected`
- **Suggested Message**: "Warning: Face mask detected. Please remove your mask to proceed."

### 3. `helmet_detected.mp3`
- **Purpose**: Played when a helmet or head covering is detected
- **Violation Type**: `helmet_detected`
- **Suggested Message**: "Warning: Helmet detected. Please remove your helmet to proceed."

### 4. `mask_and_helmet.mp3`
- **Purpose**: Played when both a mask and helmet are detected simultaneously
- **Violation Type**: `mask_and_helmet_detected`
- **Suggested Message**: "Warning: Mask and helmet detected. Please remove all face coverings to proceed."

## Audio File Format Guidelines

### Technical Specifications
- **Format**: MP3 (MPEG-1 Audio Layer 3)
- **Bitrate**: 128 kbps or higher recommended
- **Sample Rate**: 44.1 kHz (CD quality) or 48 kHz
- **Channels**: Mono or Stereo (Mono recommended for smaller file size)
- **Duration**: 3-5 seconds recommended (maximum 10 seconds)

### Quality Guidelines
- Use clear, professional voice recordings
- Ensure consistent volume levels across all files
- Avoid background noise or distortion
- Use a neutral, authoritative tone
- Speak at a moderate pace for clarity

## Creating Audio Files

### Option 1: Text-to-Speech Services
You can use online TTS services to generate audio files:
- Google Cloud Text-to-Speech
- Amazon Polly
- Microsoft Azure Speech Service
- Natural Reader
- TTSMaker (free option)

### Option 2: Professional Recording
Record audio files using:
- Professional voice talent
- High-quality microphone
- Audio editing software (Audacity, Adobe Audition, etc.)

### Option 3: Sample Files
The repository includes sample audio files with basic warnings. For production use, replace these with professionally recorded messages.

## Installation

1. Place all 4 MP3 files in the `audio/` directory
2. Ensure file names match exactly (case-sensitive on Linux/macOS)
3. Verify files are readable by the application user
4. Test audio playback before deployment

## Troubleshooting

### Audio Not Playing
- Verify all 4 MP3 files exist in the audio directory
- Check file permissions (files must be readable)
- Ensure pygame mixer is properly initialized
- Verify system audio output is working
- Check audio configuration in `config.json`

### Audio Quality Issues
- Increase bitrate to 192 kbps or 320 kbps
- Use 48 kHz sample rate
- Re-encode files with consistent settings
- Remove any silence at beginning/end of files

### Rate Limiting
The system enforces a minimum 8-second interval between audio alerts to prevent spam. This is configured in `config.json` under `audio.rate_limit_seconds`.

## Configuration

Audio settings can be modified in `config.json`:

```json
"audio": {
  "enabled": true,
  "rate_limit_seconds": 8,
  "audio_directory": "audio"
}
```

- **enabled**: Set to `false` to disable audio alerts
- **rate_limit_seconds**: Minimum seconds between alerts (default: 8)
- **audio_directory**: Path to audio files directory (default: "audio")
