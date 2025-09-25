# src/evaluate.py
"""
Load saved model and test split, produce metrics and a small report.
"""
import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "stroke_dnn.h5")
TESTDATA_PATH = os.path.join(MODELS_DIR, "test_data.npz")

def main():
    # load model and test data
    model = tf.keras.models.load_model(MODEL_PATH)
    data = np.load(TESTDATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # predict probabilities and labels (default threshold = 0.5)
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}\n")

    print("Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    # also print precision/recall/f1 with support
    prfs = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    print("\nPrecision, Recall, F1, Support:")
    print(prfs)

if __name__ == "__main__":
    main()
