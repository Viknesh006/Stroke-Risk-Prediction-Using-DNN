#!/usr/bin/env python3
"""
Desktop GUI for Stroke Risk Prediction using tkinter
"""
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np

class StrokeRiskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stroke Risk Prediction Tool")
        self.root.geometry("600x700")
        
        # Load preprocessor and model
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self):
        try:
            self.preproc = joblib.load("models/preprocessor.pkl")
            print("Preprocessor loaded successfully")
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            self.preproc = None
            
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model("models/stroke_dnn.h5")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Stroke Risk Prediction", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Input fields
        self.create_input_fields(main_frame)
        
        # Predict button
        predict_btn = ttk.Button(main_frame, text="Predict Stroke Risk", 
                               command=self.predict_risk, style="Accent.TButton")
        predict_btn.pack(pady=20)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(main_frame, text="Prediction Result")
        self.result_frame.pack(fill="x", pady=10)
        
        self.result_label = tk.Label(self.result_frame, text="Enter patient data and click predict", 
                                   font=("Arial", 12))
        self.result_label.pack(pady=20)
        
    def create_input_fields(self, parent):
        # Create input fields
        fields_frame = ttk.Frame(parent)
        fields_frame.pack(fill="both", expand=True)
        
        # Left column
        left_frame = ttk.Frame(fields_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Right column  
        right_frame = ttk.Frame(fields_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.vars = {}
        
        # Left column fields
        self.add_field(left_frame, "Age", "age", "float")
        self.add_field(left_frame, "Average Glucose Level", "avg_glucose_level", "float")
        self.add_field(left_frame, "BMI", "bmi", "float")
        self.add_field(left_frame, "Gender", "gender", "combo", ["Male", "Female", "Other"])
        self.add_field(left_frame, "Ever Married", "ever_married", "combo", ["Yes", "No"])
        
        # Right column fields
        self.add_field(right_frame, "Hypertension", "hypertension", "combo", ["0", "1"])
        self.add_field(right_frame, "Heart Disease", "heart_disease", "combo", ["0", "1"])
        self.add_field(right_frame, "Work Type", "work_type", "combo", 
                      ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        self.add_field(right_frame, "Residence Type", "Residence_type", "combo", ["Urban", "Rural"])
        self.add_field(right_frame, "Smoking Status", "smoking_status", "combo", 
                      ["formerly smoked", "never smoked", "smokes", "Unknown"])
        
    def add_field(self, parent, label, var_name, field_type, options=None):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        ttk.Label(frame, text=label + ":").pack(anchor="w")
        
        if field_type == "combo":
            var = tk.StringVar()
            combo = ttk.Combobox(frame, textvariable=var, values=options, state="readonly")
            combo.pack(fill="x")
            if options:
                combo.set(options[0])
        else:
            var = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(fill="x")
            
        self.vars[var_name] = var
        
    def predict_risk(self):
        try:
            # Get input data
            data = {}
            for key, var in self.vars.items():
                value = var.get()
                if not value:
                    messagebox.showerror("Error", f"Please fill in {key}")
                    return
                    
                # Convert numeric fields
                if key in ["age", "avg_glucose_level", "bmi"]:
                    try:
                        data[key] = float(value)
                    except ValueError:
                        messagebox.showerror("Error", f"{key} must be a number")
                        return
                elif key in ["hypertension", "heart_disease"]:
                    data[key] = int(value)
                else:
                    data[key] = value
            
            # Make prediction
            if self.preproc is None:
                messagebox.showerror("Error", "Preprocessor not loaded")
                return
                
            df = pd.DataFrame([data])
            X = self.preproc.transform(df)
            
            if self.model is not None:
                prediction = self.model.predict(X, verbose=0)
                risk = float(prediction[0][0])
                method = "Neural Network Model"
            else:
                # Fallback calculation
                risk = self.calculate_simple_risk(data)
                method = "Heuristic Method (Model unavailable)"
            
            # Display result
            risk_percent = risk * 100
            risk_level = "HIGH" if risk > 0.5 else "MODERATE" if risk > 0.3 else "LOW"
            
            result_text = f"""
Stroke Risk Prediction Result:

Risk Probability: {risk_percent:.1f}%
Risk Level: {risk_level}
Method: {method}

Recommendation: 
{'Consult a doctor immediately' if risk > 0.7 else 
 'Regular health monitoring recommended' if risk > 0.4 else 
 'Maintain healthy lifestyle'}
"""
            
            self.result_label.config(text=result_text, 
                                   fg="red" if risk > 0.5 else "orange" if risk > 0.3 else "green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def calculate_simple_risk(self, data):
        """Simple risk calculation fallback"""
        risk = 0.1
        
        if data["age"] > 65: risk += 0.3
        elif data["age"] > 50: risk += 0.2
        elif data["age"] > 35: risk += 0.1
        
        if data["hypertension"] == 1: risk += 0.2
        if data["heart_disease"] == 1: risk += 0.25
        
        if data["avg_glucose_level"] > 200: risk += 0.15
        elif data["avg_glucose_level"] > 140: risk += 0.1
        
        if data["bmi"] > 30: risk += 0.1
        elif data["bmi"] > 25: risk += 0.05
        
        if data["smoking_status"] in ["smokes", "formerly smoked"]: risk += 0.15
        
        return min(risk, 0.95)

if __name__ == "__main__":
    root = tk.Tk()
    app = StrokeRiskGUI(root)
    root.mainloop()
