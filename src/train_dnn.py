# src/train_dnn.py
"""
Train a simple MLP (DNN) for stroke prediction using the preprocessor saved
by preprocess.py. Saves trained model and test-split for evaluation.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
PREPROC_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "stroke_dnn.h5")
TESTDATA_PATH = os.path.join(MODELS_DIR, "test_data.npz")

def load_data(csv_path="data/healthcare-dataset-stroke-data.csv"):
    df = pd.read_csv(csv_path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    X = df.drop(columns=["stroke"])
    y = df["stroke"].astype(int)
    return X, y

def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

def main():
    # load raw data
    X, y = load_data()

    # load preprocessor
    preproc = joblib.load(PREPROC_PATH)

    # preprocess full dataset then split (to avoid information leak from test)
    X_proc = preproc.transform(X)

    # train-test split (stratify to preserve class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    # handle imbalance on training set using SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # build, train
    model = build_model(X_train_res.shape[1])

    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=4),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_auc", mode="max")
    ]

    history = model.fit(
        X_train_res, y_train_res,
        validation_split=0.15,
        epochs=100,
        batch_size=128,
        callbacks=cb,
        verbose=2
    )

    # save final model (best saved by ModelCheckpoint)
    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # save test set for evaluation
    np.savez(TESTDATA_PATH, X_test=X_test, y_test=y_test)
    print(f"Test split saved to: {TESTDATA_PATH}")

if __name__ == "__main__":
    main()
