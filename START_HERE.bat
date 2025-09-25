@echo off
echo ========================================
echo    STROKE RISK PREDICTION - QUICK START
echo ========================================
echo.
echo Choose your preferred method:
echo.
echo 1. Web API (Recommended)
echo 2. Desktop GUI  
echo 3. Quick Test
echo 4. View Documentation
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto webapi
if "%choice%"=="2" goto gui
if "%choice%"=="3" goto test
if "%choice%"=="4" goto docs
goto invalid

:webapi
echo.
echo Starting Web API server...
echo API will be available at: http://localhost:8001
echo Press Ctrl+C to stop the server
echo.
".\venv\Scripts\python.exe" -m uvicorn src.app_simple:app --reload --host 0.0.0.0 --port 8001
goto end

:gui
echo.
echo Starting Desktop GUI...
".\venv\Scripts\python.exe" stroke_gui.py
goto end

:test
echo.
echo Running quick test...
".\venv\Scripts\python.exe" test_model.py
echo.
pause
goto end

:docs
echo.
echo Opening documentation...
start QUICK_START.md
start README.md
goto end

:invalid
echo Invalid choice. Please run the script again.
pause
goto end

:end
echo.
echo Thank you for using Stroke Risk Prediction!
pause
