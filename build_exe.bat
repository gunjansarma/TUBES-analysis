@echo off
REM ============================================================================
REM Builds SensorStreamMonitor.exe from app.py + launcher.py.
REM MUST be run on a real Windows machine — PyInstaller cannot cross-compile,
REM so running this on Mac/Linux produces a Mac/Linux binary, not a .exe.
REM
REM Prerequisites (one-time, on the Windows machine that runs this):
REM   1. Install Python 3.10+ from python.org (check "Add to PATH" during install)
REM   2. Open Command Prompt in this folder and run:
REM        pip install -r requirements-build.txt
REM ============================================================================

pyinstaller --onedir --name SensorStreamMonitor ^
  --collect-all streamlit ^
  --collect-all plotly ^
  --collect-all kaleido ^
  --copy-metadata streamlit ^
  --copy-metadata pandas ^
  --copy-metadata numpy ^
  --copy-metadata scipy ^
  --copy-metadata plotly ^
  --copy-metadata kaleido ^
  --add-data "app.py;." ^
  --noconfirm ^
  launcher.py

echo.
echo ============================================================================
echo Build finished. The app is in: dist\SensorStreamMonitor\
echo Run it by double-clicking: dist\SensorStreamMonitor\SensorStreamMonitor.exe
echo To share with colleagues, zip the WHOLE "SensorStreamMonitor" folder —
echo the .exe alone will not run without the other files next to it.
echo ============================================================================
pause