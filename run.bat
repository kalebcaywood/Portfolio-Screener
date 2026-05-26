@echo off
title Quant Portfolio Lab
cd /d "%~dp0"

REM ─── Check if server is already running on port 8501 ───────────────────────
netstat -an | findstr ":8501.*LISTENING" >nul
if %errorlevel% == 0 (
    echo Quant Portfolio Lab is already running.
    echo Opening browser...
    start "" http://localhost:8501
    timeout /t 2 /nobreak >nul
    exit /b 0
)

REM ─── Verify venv exists ────────────────────────────────────────────────────
if not exist ".venv\Scripts\streamlit.exe" (
    echo ERROR: Virtual environment not found at .venv\
    echo.
    echo First-time setup needed. Run these commands once:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ─── Schedule browser to open after server starts ──────────────────────────
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:8501"

REM ─── Launch streamlit (foreground; closing this window stops the server) ──
echo ============================================================
echo   Quant Portfolio Lab
echo ============================================================
echo   URL:    http://localhost:8501
echo   Status: Starting up... your browser will open shortly
echo.
echo   Close this window to stop the server.
echo ============================================================
echo.

.venv\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501
