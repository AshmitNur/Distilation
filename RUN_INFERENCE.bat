@echo off
chcp 65001 >nul
title Bangla Distill — Local GPU Inference Engine
color 0a

echo ======================================================================
echo           Bangla Distill - Local GPU Inference CLI
echo ======================================================================
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

".venv\Scripts\python.exe" scripts\run_local_inference.py --interactive %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Inference session exited. Press any key to close...
    pause >nul
)
