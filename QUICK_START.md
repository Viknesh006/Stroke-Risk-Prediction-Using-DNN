# 🚀 QUICK START - Stroke Risk Prediction

## Every Time You Open This Project

### Method 1: Web API (Most Popular)
1. **Open PowerShell** in project folder
2. **Run this command:**
   ```powershell
   & ".\venv\Scripts\python.exe" -m uvicorn src.app_simple:app --reload --host 0.0.0.0 --port 8001
   ```
3. **Open browser:** http://localhost:8001
4. **Test API:** Run `& ".\venv\Scripts\python.exe" test_api.py`

### Method 2: Desktop GUI (User-Friendly)
1. **Open PowerShell** in project folder
2. **Run this command:**
   ```powershell
   & ".\venv\Scripts\python.exe" stroke_gui.py
   ```
3. **Use the GUI** to input patient data and get predictions

### Method 3: Quick Test
1. **Test the model:**
   ```powershell
   & ".\venv\Scripts\python.exe" test_model.py
   ```

## 📋 What Each Method Does

| Method | Best For | Output |
|--------|----------|---------|
| **Web API** | Integration, multiple users | JSON responses via HTTP |
| **Desktop GUI** | Single user, easy interface | Visual desktop application |
| **Quick Test** | Verification, debugging | Console output with predictions |

## 🔍 Verification Steps

### 1. Check if everything is working:
```powershell
# Test the model directly
& ".\venv\Scripts\python.exe" test_model.py

# Should show:
# SUCCESS: Preprocessor loaded successfully
# SUCCESS: Model loaded successfully  
# SUCCESS: Prediction works: 0.5619
```

### 2. Check API health (if using Web API):
- Visit: http://localhost:8001/health
- Should show: `"status": "healthy"`

### 3. Make a test prediction:
```powershell
& ".\venv\Scripts\python.exe" test_api.py
```

## ⚡ One-Click Solutions

### For Windows Users:
- **Double-click** `run_app.bat` (runs full pipeline)
- **Or run** `.\run_app.ps1` in PowerShell

## 🎯 Sample Input Data

```json
{
  "gender": "Male",
  "age": 67.0,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes", 
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

**Expected Output:** ~53-56% stroke risk

## 🚨 If Something Goes Wrong

1. **Virtual environment issues:**
   ```powershell
   # Check if venv exists
   ls venv\Scripts\
   # Should see python.exe and pip.exe
   ```

2. **Missing packages:**
   ```powershell
   & ".\venv\Scripts\pip.exe" install -r requirements.txt
   ```

3. **Model not found:**
   ```powershell
   # Check models folder
   ls models\
   # Should see: preprocessor.pkl, stroke_dnn.h5
   ```

4. **Port already in use:**
   - Change port: `--port 8002` instead of `--port 8001`
   - Or use the GUI method instead

## 📞 Quick Commands Reference

```powershell
# Start Web API
& ".\venv\Scripts\python.exe" -m uvicorn src.app_simple:app --reload --host 0.0.0.0 --port 8001

# Start GUI
& ".\venv\Scripts\python.exe" stroke_gui.py

# Test Model
& ".\venv\Scripts\python.exe" test_model.py

# Test API
& ".\venv\Scripts\python.exe" test_api.py

# Health Check
curl http://localhost:8001/health
```

---
**💡 Tip:** Bookmark this file for quick reference every time you open the project!
