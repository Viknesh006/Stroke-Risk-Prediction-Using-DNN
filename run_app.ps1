Write-Host "Starting Stroke Risk Prediction API..." -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Creating preprocessor..." -ForegroundColor Yellow
& ".\venv\Scripts\python.exe" src\preprocess.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in preprocessing step!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Training the model..." -ForegroundColor Yellow
& ".\venv\Scripts\python.exe" src\train_dnn.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in training step!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Step 3: Starting FastAPI server..." -ForegroundColor Yellow
Write-Host "The API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
& ".\venv\Scripts\python.exe" -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
