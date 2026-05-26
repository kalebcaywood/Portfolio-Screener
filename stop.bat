@echo off
title Stop Quant Portfolio Lab
echo Stopping any Quant Portfolio Lab server on port 8501...
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501.*LISTENING"') do (
    echo Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

REM Also clean up any stray streamlit processes
taskkill /F /IM streamlit.exe >nul 2>&1

echo Done.
timeout /t 2 /nobreak >nul
