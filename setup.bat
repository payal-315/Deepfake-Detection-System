@echo off
title Deepfake Detection - Initial Setup

echo ==========================================
echo         Project Initial Setup
echo ==========================================
echo.

REM ------------------------------
REM CHECK wkhtmltopdf
REM ------------------------------
echo Checking for wkhtmltopdf...

where wkhtmltopdf >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] wkhtmltopdf NOT FOUND!
    echo PDF generation will NOT work without it.
    echo.
    set /p CHOICE="Do you want to open the download page? (y/n): "
    if /I "%CHOICE%"=="y" (
        start https://wkhtmltopdf.org/downloads.html
    )
) ELSE (
    echo wkhtmltopdf found.
)
echo.

REM ------------------------------
REM SETUP PYTHON VENV
REM ------------------------------
echo Checking backend virtual environment...

if NOT exist backend\venv (
    echo Creating virtual environment...
    python -m venv backend\venv
) ELSE (
    echo venv already exists.
)

echo Installing backend dependencies...
call backend\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
deactivate

echo Backend setup complete.
echo.

REM ------------------------------
REM SETUP FRONTEND
REM ------------------------------
echo Installing frontend dependencies...
cd frontend
npm install
cd ..

echo Frontend setup complete.
echo.

echo ==========================================
echo Setup finished successfully!
echo You can now run: start.bat
echo ==========================================
echo.

pause
