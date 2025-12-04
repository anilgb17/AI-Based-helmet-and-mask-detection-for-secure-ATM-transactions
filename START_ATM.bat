@echo off
REM ============================================================================
REM ATM Security System - Windows Launcher
REM ============================================================================
REM This batch file starts the ATM Security System on Windows
REM Double-click this file to run the application
REM ============================================================================

echo.
echo ================================================================================
echo ATM SECURITY SYSTEM - WINDOWS LAUNCHER
echo ================================================================================
echo.
echo Starting the ATM Security System...
echo.
echo Once started, open your browser and go to: http://127.0.0.1:5004
echo.
echo Press Ctrl+C to stop the application
echo.
echo ================================================================================
echo.

REM Run the launcher script
python run_atm_system.py

REM If Python command fails, try python3
if errorlevel 1 (
    echo.
    echo Python command failed. Trying python3...
    python3 run_atm_system.py
)

REM Pause to see any error messages
echo.
echo.
echo ================================================================================
echo Application stopped.
echo ================================================================================
pause
