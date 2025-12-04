"""
================================================================================
ATM SECURITY SYSTEM - LAUNCHER FOR IDLE AND OTHER ENVIRONMENTS
================================================================================

This launcher script ensures the application runs correctly in IDLE and other
Python environments by properly setting up the Python path.

Usage:
    1. Open this file in IDLE
    2. Press F5 or Run > Run Module
    3. Or run from command line: python run_atm_system.py

Author: ATM Security System Development Team
Version: 1.0
Date: November 19, 2025
================================================================================
"""

# ============================================================================
# FIX PYTHON PATH FOR IDLE AND OTHER ENVIRONMENTS
# ============================================================================
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to Python path if not already there
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added to Python path: {script_dir}")

# Change working directory to script directory
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

# ============================================================================
# NOW IMPORT AND RUN THE MAIN APPLICATION
# ============================================================================

print("\n" + "="*80)
print("ATM SECURITY SYSTEM - STARTING")
print("="*80)
print("Initializing modules...")
print("="*80 + "\n")

# Import the main application
from app_complete_commented import main

# Run the application
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
