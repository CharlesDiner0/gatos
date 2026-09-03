@echo off
title Detector de Gestos - Futbolistas
echo Iniciando detector de gestos de futbolistas...
cd /d "C:\Users\cxrlo\Desktop\gatos-main"
".venv\Scripts\python.exe" gesture_meme.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ocurrio un error al ejecutar el programa.
    pause
)
