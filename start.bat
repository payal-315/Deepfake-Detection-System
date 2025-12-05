@echo off
title Deepfake Detection - Start

echo ==========================================
echo      Starting Deepfake Detection App
echo ==========================================
echo.

REM ------------------------------
REM START BACKEND
REM ------------------------------
echo Starting backend...
start cmd /k "cd backend && call venv\Scripts\activate && python main.py"
echo Backend launched.
echo.

REM ------------------------------
REM START FRONTEND
REM ------------------------------
echo Starting frontend...
start cmd /k "cd frontend && npm run dev -- --port 5555"
echo Frontend launched.
echo.

echo ==========================================
echo  All services started successfully!
echo  Backend:  http://127.0.0.1:8000
echo  Frontend: http://localhost:5555
echo ==========================================
echo.

pause
