#!/usr/bin/env python3
"""
Test script to verify the API is working
"""
import requests
import json

# Test data
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

print("Testing Stroke Risk Prediction API...")
print("=" * 50)

try:
    # Test health endpoint
    health_response = requests.get("http://localhost:8001/health")
    print("Health Check:")
    print(json.dumps(health_response.json(), indent=2))
    print()
    
    # Test prediction endpoint
    print("Making prediction...")
    response = requests.post("http://localhost:8001/predict", json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print("SUCCESS! Prediction result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"ERROR: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to API server.")
    print("Make sure the server is running on http://localhost:8001")
except Exception as e:
    print(f"ERROR: {e}")
