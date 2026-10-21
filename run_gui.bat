@echo off
REM Launch the registration GUI with the Anaconda base interpreter
REM (has cv2, h5py, skimage, pandas, matplotlib, tkinter).
set PYEXE=C:\Users\b296295\AppData\Local\anaconda3\python.exe
cd /d "%~dp0"
"%PYEXE%" xrf_registration_gui.py %*
if errorlevel 1 pause
