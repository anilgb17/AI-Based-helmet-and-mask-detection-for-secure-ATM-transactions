"""
Test Audio System
Verifies audio files and playback functionality
"""
import time
from audio.audio_alert_system import AudioAlertSystem

print("=" * 70)
print("AUDIO SYSTEM TEST")
print("=" * 70)

# Initialize audio system
print("\n[1/3] Initializing audio system...")
try:
    audio = AudioAlertSystem('audio')
    print("✓ Audio system initialized")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    exit(1)

# Test welcome message
print("\n[2/3] Testing welcome message...")
try:
    print("Playing welcome.mp3...")
    audio.play_system_message('welcome')
    print("✓ Welcome message queued")
    print("  (Wait 3-5 seconds for audio to play)")
    time.sleep(5)
except Exception as e:
    print(f"✗ Error playing welcome: {e}")

# Test transaction complete message
print("\n[3/3] Testing transaction complete message...")
try:
    print("Playing transaction_complete.mp3...")
    audio.play_system_message('transaction_complete')
    print("✓ Transaction complete message queued")
    print("  (Wait 3-5 seconds for audio to play)")
    time.sleep(5)
except Exception as e:
    print(f"✗ Error playing transaction complete: {e}")

# Test violation alerts
print("\n[BONUS] Testing violation alerts...")
print("Testing mask_detected.mp3...")
audio.queue_alert('mask_detected')
print("  (Wait 3-5 seconds for audio to play)")
time.sleep(5)

# Cleanup
print("\n[CLEANUP] Shutting down audio system...")
audio.shutdown()
print("✓ Audio system shutdown complete")

print("\n" + "=" * 70)
print("AUDIO TEST COMPLETE")
print("=" * 70)
print("\nIf you heard audio, the system is working correctly!")
print("If no audio:")
print("  1. Check volume is not muted")
print("  2. Check audio files exist in audio/ directory")
print("  3. Check pygame is installed: pip install pygame-ce")
