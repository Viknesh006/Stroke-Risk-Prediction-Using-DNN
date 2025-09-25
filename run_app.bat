@echo off
echo Starting Stroke Risk Prediction API...
echo.
echo Step 1: Creating preprocessor...
".\venv\Scripts\python.exe" src\preprocess.py
if %errorlevel% neq 0 (
    echo Error in preprocessing step!
    pause
    exit /b 1
)

echo.
echo Step 2: Training the model...
".\venv\Scripts\python.exe" src\train_dnn.py
if %errorlevel% neq 0 (
    echo Error in training step!
    pause
    exit /b 1
)

echo.
echo Step 3: Starting FastAPI server...
echo The API will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
".\venv\Scripts\python.exe" -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
