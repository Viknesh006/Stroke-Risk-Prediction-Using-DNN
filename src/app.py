# src/app.py
"""
Simple FastAPI app that loads the preprocessor + model and exposes one /predict endpoint.
Send a JSON with the *raw* feature names (same as the CSV header except 'id' and 'stroke').
Example:
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
"""
import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Stroke Risk Prediction API")

MODELS_DIR = "models"
PREPROC_PATH = f"{MODELS_DIR}/preprocessor.pkl"
MODEL_PATH = f"{MODELS_DIR}/stroke_dnn.h5"

preproc = joblib.load(PREPROC_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

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

@app.post("/predict")
def predict(payload: InputData):
    df = pd.DataFrame([payload.dict()])
    X = preproc.transform(df)
    proba = float(model.predict(X).ravel()[0])
    return {"stroke_risk_probability": proba}
