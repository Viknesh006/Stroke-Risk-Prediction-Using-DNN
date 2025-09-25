#!/usr/bin/env python3
"""
Simple test script to verify the trained model works
"""
import os
import joblib
import pandas as pd
import numpy as np

print("=== Testing Stroke Risk Prediction Model ===")

# Test if we can load the preprocessor
try:
    preproc = joblib.load("models/preprocessor.pkl")
    print("SUCCESS: Preprocessor loaded successfully")
    
    # Create test data
    test_data = {
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
    
    df = pd.DataFrame([test_data])
    X = preproc.transform(df)
    print("SUCCESS: Data preprocessing works")
    print(f"Preprocessed data shape: {X.shape}")
    
except Exception as e:
    print(f"ERROR: Problem with preprocessor: {e}")

# Test if we can load the model
try:
    import tensorflow as tf
    model = tf.keras.models.load_model("models/stroke_dnn.h5")
    print("SUCCESS: Model loaded successfully")
    
    # Make a prediction
    if 'X' in locals():
        prediction = model.predict(X)
        print(f"SUCCESS: Prediction works: {float(prediction[0][0]):.4f}")
        print(f"Stroke risk probability: {float(prediction[0][0]):.2%}")
    
except Exception as e:
    print(f"ERROR: Problem with TensorFlow model: {e}")
    print("This might be a TensorFlow installation issue on Windows")
