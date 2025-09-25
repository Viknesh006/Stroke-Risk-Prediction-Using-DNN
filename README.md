# Stroke Risk Prediction - Deep Neural Network

A machine learning project that predicts stroke risk using patient health data with a Deep Neural Network (DNN).

## 🚀 Quick Start Guide

### Step 1: Open Project
1. Open this folder in your IDE: `d:\NNW LAP\Stroke Risk Prediction - DNN`
2. Open PowerShell/Terminal in this directory

### Step 2: Choose Your Method

#### Option A: Web API (Recommended)
```powershell
& ".\venv\Scripts\python.exe" -m uvicorn src.app_simple:app --reload --host 0.0.0.0 --port 8001
├── data/                   # Dataset
│   └── healthcare-dataset-stroke-data.csv
├── models/                 # Trained models
│   ├── preprocessor.pkl    # Data preprocessor
│   ├── stroke_dnn.h5       # Neural network model
│   └── test_data.npz       # Test dataset
├── src/                    # Source code
│   ├── app.py              # Original FastAPI app
│   ├── app_simple.py       # Simplified API (recommended)
│   ├── preprocess.py       # Data preprocessing
│   ├── train_dnn.py        # Model training
│   └── evaluate.py         # Model evaluation
├── venv/                   # Virtual environment
├── stroke_gui.py           # Desktop GUI application
├── test_model.py           # Model testing script
├── test_api.py             # API testing script
├── run_app.bat             # Windows batch script
├── run_app.ps1             # PowerShell script
└── requirements.txt        # Python dependencies
