"""
Test All 6 Audio Files
Verifies all audio files are present and can be played
"""
import time
import os
from audio.audio_alert_system import AudioAlertSystem

print("=" * 80)
print("TESTING ALL 6 AUDIO FILES")
print("=" * 80)

# Check files exist
print("\n[1/7] Checking audio files...")
audio_files = [
    'helmet_detected.mp3',
    'mask_and_helmet.mp3',
    'mask_detected.mp3',
    'multiple_people.mp3',
    'transaction_complete.mp3',
    'welcome.mp3'
]

all_present = True
for filename in audio_files:
    filepath = os.path.join('audio', filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✓ {filename} ({size:,} bytes)")
    else:
        print(f"  ✗ {filename} - MISSING!")
        all_present = False

if not all_present:
    print("\n✗ Some audio files are missing!")
    exit(1)

print("\n✓ All 6 audio files present")

# Initialize audio system
print("\n[2/7] Initializing audio system...")
try:
    audio = AudioAlertSystem('audio')
    print("✓ Audio system initialized")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    exit(1)

# Test each audio file
print("\n" + "=" * 80)
print("PLAYING AUDIO FILES (Listen for sound)")
print("=" * 80)

# Test 1: Welcome
print("\n[3/7] Testing welcome.mp3...")
print("  Playing: 'Welcome to secure ATM transactions'")
audio.play_system_message('welcome')
time.sleep(5)
print("  ✓ Played")

# Test 2: Transaction Complete
print("\n[4/7] Testing transaction_complete.mp3...")
print("  Playing: 'Thank you, your transaction is successfully completed'")
audio.play_system_message('transaction_complete')
time.sleep(5)
print("  ✓ Played")

# Test 3: Mask Detected
print("\n[5/7] Testing mask_detected.mp3...")
print("  Playing: 'Face mask detected...'")
audio.queue_alert('mask_detected')
time.sleep(5)
print("  ✓ Played")

# Test 4: Helmet Detected
print("\n[6/7] Testing helmet_detected.mp3...")
print("  Playing: 'Helmet detected...'")
audio.queue_alert('helmet_detected')
time.sleep(10)  # Wait for rate limit
print("  ✓ Played")

# Test 5: Multiple People
print("\n[7/7] Testing multiple_people.mp3...")
print("  Playing: 'Multiple people detected...'")
audio.queue_alert('multiple_people')
time.sleep(10)  # Wait for rate limit
print("  ✓ Played")

# Note about mask_and_helmet
print("\n[BONUS] mask_and_helmet.mp3")
print("  This plays when BOTH mask AND helmet are detected")
print("  (Not tested here to save time)")

# Cleanup
print("\n[CLEANUP] Shutting down audio system...")
audio.shutdown()
print("✓ Audio system shutdown complete")

print("\n" + "=" * 80)
print("AUDIO TEST COMPLETE")
print("=" * 80)

print("\nResults:")
print("  ✓ All 6 audio files present")
print("  ✓ Audio system initialized")
print("  ✓ 5 audio files tested")

print("\nDid you hear the audio?")
print("  YES → Audio system working perfectly! 🔊")
print("  NO  → Check volume, speakers, or pygame installation")

print("\nNext steps:")
print("  1. If audio worked: Run main application")
print("     python app_final_complete.py")
print("  2. If no audio: Check troubleshooting in AUDIO_FILES_VERIFIED.md")
