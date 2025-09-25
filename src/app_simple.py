# src/app_simple.py
"""
Simplified FastAPI app with better TensorFlow error handling
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stroke Risk Prediction API - Simple Version")

# Global variables for model components
preproc = None
model = None

# Load preprocessor at startup
try:
    MODELS_DIR = "models"
    PREPROC_PATH = f"{MODELS_DIR}/preprocessor.pkl"
    preproc = joblib.load(PREPROC_PATH)
    logger.info("Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessor: {e}")

# Try to load TensorFlow model with better error handling
try:
    import tensorflow as tf
    # Set TensorFlow to use CPU only to avoid DLL issues
    tf.config.set_visible_devices([], 'GPU')
    
    MODEL_PATH = f"{MODELS_DIR}/stroke_dnn.h5"
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("TensorFlow model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TensorFlow model: {e}")
    logger.info("API will run in preprocessor-only mode")

class InputData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/")
def root():
    return {
        "message": "Stroke Risk Prediction API",
        "status": "running",
        "preprocessor_loaded": preproc is not None,
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "preprocessor": "loaded" if preproc else "failed",
        "model": "loaded" if model else "failed"
    }

@app.post("/predict")
def predict(payload: InputData):
    if preproc is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")
    
    try:
        # Convert to DataFrame and preprocess
        df = pd.DataFrame([payload.dict()])
        X = preproc.transform(df)
        
        if model is not None:
            # Make prediction with TensorFlow model
            proba = float(model.predict(X, verbose=0).ravel()[0])
            return {
                "stroke_risk_probability": proba,
                "risk_percentage": f"{proba:.2%}",
                "method": "tensorflow_model"
            }
        else:
            # Fallback: Simple risk calculation based on known risk factors
            risk_score = calculate_simple_risk(payload)
            return {
                "stroke_risk_probability": risk_score,
                "risk_percentage": f"{risk_score:.2%}",
                "method": "simple_heuristic",
                "note": "TensorFlow model unavailable, using heuristic approach"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def calculate_simple_risk(data: InputData):
    """
    Simple heuristic risk calculation when TensorFlow model is unavailable
    """
    risk = 0.1  # Base risk
    
    # Age factor
    if data.age > 65:
        risk += 0.3
    elif data.age > 50:
        risk += 0.2
    elif data.age > 35:
        risk += 0.1
    
    # Health conditions
    if data.hypertension == 1:
        risk += 0.2
    if data.heart_disease == 1:
        risk += 0.25
    
    # Glucose level
    if data.avg_glucose_level > 200:
        risk += 0.15
    elif data.avg_glucose_level > 140:
        risk += 0.1
    
    # BMI
    if data.bmi > 30:
        risk += 0.1
    elif data.bmi > 25:
        risk += 0.05
    
    # Smoking
    if data.smoking_status in ["smokes", "formerly smoked"]:
        risk += 0.15
    
    # Cap at 95%
    return min(risk, 0.95)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
