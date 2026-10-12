@echo off
chcp 65001 >nul
title Bangla Distill — AI Model Arena & Chat UI
color 0b

echo ======================================================================
echo           Bangla Distill - AI Model Arena and Chat UI
echo ======================================================================
echo.
echo [1/2] Loading Qwen2.5-0.5B + Bangla LoRA Adapter on NVIDIA GPU...
echo [2/2] Opening browser at http://localhost:8000
echo.
echo ----------------------------------------------------------------------
echo  * Side-by-Side Arena (Vanilla Base vs. Distilled Student)
echo  * Interactive Bengali Chat with LoRA toggle
echo  * Live RTX 4070 VRAM and Speed Telemetry
echo ----------------------------------------------------------------------
echo.
echo To STOP the server, simply close this window or press Ctrl+C.
echo ======================================================================
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:: Automatically open browser in 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"

:: Start Uvicorn Server with GPU Model Engine
".venv\Scripts\python.exe" -m uvicorn app:app --app-dir ui --host 127.0.0.1 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server encountered an error. Press any key to exit...
    pause >nul
)
