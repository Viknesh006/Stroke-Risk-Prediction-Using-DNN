# src/preprocess.py
"""
Builds and saves a preprocessing pipeline for the Kaggle
'healthcare-dataset-stroke-data.csv' dataset.
"""
import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

def build_preprocessor():
    # numeric and categorical columns for the Kaggle stroke dataset
    num_cols = ["age", "avg_glucose_level", "bmi"]
    cat_cols = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    # numeric pipeline: median imputation + standard scaling
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # categorical pipeline: use one-hot encoding, ignore unknown categories
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    return preproc

def main():
    # assumes CSV is at data/healthcare-dataset-stroke-data.csv (project root)
    df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
    # drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=["stroke"])
    preproc = build_preprocessor()
    preproc.fit(X)

    joblib.dump(preproc, PREPROCESSOR_PATH)
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    main()
