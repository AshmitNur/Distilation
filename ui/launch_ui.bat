@echo off
title Bangla Distill Web UI
echo ======================================================================
echo Launching Bangla Distill Web UI on RTX 4070 GPU...
echo ======================================================================
echo.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

start http://localhost:8000
"..\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
