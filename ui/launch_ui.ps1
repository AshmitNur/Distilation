Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Launching Bangla Distill Web UI on RTX 4070 GPU..." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

$env:PYTHONIOENCODING = "utf-8"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$parentDir = Split-Path -Parent $scriptDir
Start-Process "http://localhost:8000"
& "$parentDir\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
