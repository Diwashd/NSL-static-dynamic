"""Evaluate all models (static + dynamic): generate reports and confusion matrices.

Usage: run from repo root:
    python scripts/evaluate_all_models.py

This script will:
- Load dataset (both static landmarks and dynamic sequences)
- Create reproducible train/test splits
- Load label encoders for each model type
- Find saved models under `models/` (static .pkl + dynamic .h5)
- Run predictions and save classification reports and confusion matrices
- Organize outputs by model type (static/dynamic)
"""
import os
import glob
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.getcwd()
if not os.path.exists(os.path.join(BASE_DIR, "dataset_word")):
    parent = os.path.abspath(os.path.join(BASE_DIR, ".."))
    if os.path.exists(os.path.join(parent, "dataset_word")):
        BASE_DIR = parent

DATA_DIR_STATIC = os.path.join(BASE_DIR, "dataset_word", "landmarks", "static")
DATA_DIR_DYNAMIC = os.path.join(BASE_DIR, "dataset_word", "landmarks", "final")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eval_reports")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DATASET LOADING
# =========================================================

def load_dataset(data_dir):
    """Load .npy files from directory structure."""
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    for cls in classes:
        path = os.path.join(data_dir, cls)
        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            if file.endswith(".npy"):
                X.append(np.load(os.path.join(path, file)))
                y.append(cls)
    
    if not X:
        return None, None, None
    
    X = np.array(X)
    y = np.array(y)
    return X, y, classes


def normalize_sequence(seq, target_length, target_dim):
    """Normalize sequence to fixed shape."""
    seq = np.asarray(seq)
    if seq.ndim != 2:
        return None
    
    if seq.shape[0] < target_length:
        pad_rows = np.zeros((target_length - seq.shape[0], seq.shape[1]), dtype=seq.dtype)
        seq = np.vstack([seq, pad_rows])
    elif seq.shape[0] > target_length:
        seq = seq[:target_length]
    
    if seq.shape[1] < target_dim:
        pad_cols = np.zeros((seq.shape[0], target_dim - seq.shape[1]), dtype=seq.dtype)
        seq = np.hstack([seq, pad_cols])
    elif seq.shape[1] > target_dim:
        seq = seq[:, :target_dim]
    
    return seq


def prepare_dynamic_dataset(data_dir):
    """Load and normalize dynamic sequences."""
    X, y, classes = load_dataset(data_dir)
    if X is None:
        return None, None, None, None
    
    raw_sequences = []
    labels = []
    
    for seq, label in zip(X, y):
        raw_sequences.append(seq)
        labels.append(label)
    
    if not raw_sequences:
        return None, None, None, None
    
    target_length = max(seq.shape[0] for seq in raw_sequences if seq.ndim == 2)
    target_dim = max(seq.shape[1] for seq in raw_sequences if seq.ndim == 2)
    
    X_normalized = []
    for seq in raw_sequences:
        normalized = normalize_sequence(seq, target_length, target_dim)
        if normalized is not None:
            X_normalized.append(normalized)
    
    return np.stack(X_normalized), np.array(labels), classes, (target_length, target_dim)


# =========================================================
# MODEL DISCOVERY AND LOADING
# =========================================================

def get_static_models():
    """Find all static sklearn models (.pkl files)."""
    models = {}
    
    # Static models in root models/
    for pkl_file in glob.glob(os.path.join(MODEL_DIR, "model_*.pkl")):
        name = os.path.basename(pkl_file)
        models[name] = pkl_file
    
    # Scaler for static models
    scalers = {}
    for scaler_file in glob.glob(os.path.join(MODEL_DIR, "scaler_*.pkl")):
        scale_name = os.path.basename(scaler_file)
        scalers[scale_name] = scaler_file
    
    return models, scalers


def get_dynamic_models():
    """Find all dynamic Keras models (.h5 files)."""
    models = {}
    
    # Final models
    for h5_file in glob.glob(os.path.join(MODEL_DIR, "final", "*.h5")):
        name = os.path.basename(h5_file)
        if name not in ["best_model.h5"]:
            models[name] = h5_file
    
    # Also include best_model.h5 if it exists
    best = os.path.join(MODEL_DIR, "final", "best_model.h5")
    if os.path.exists(best):
        models["best_model.h5"] = best
    
    # Dynamic models
    for h5_file in glob.glob(os.path.join(MODEL_DIR, "dynamic", "*.h5")):
        name = os.path.basename(h5_file)
        models[name] = h5_file
    
    return models


# =========================================================
# PREDICTION AND EVALUATION
# =========================================================

def predict_static(model_path, X_test, scaler=None):
    """Load sklearn model and make predictions."""
    model = joblib.load(model_path)
    
    # Flatten and scale if scaler provided
    X_flat = X_test.reshape((X_test.shape[0], -1))
    
    if scaler is not None:
        X_flat = scaler.transform(X_flat)
    
    pred = model.predict(X_flat)
    return np.array(pred)


def predict_dynamic(model_path, X_test):
    """Load Keras model and make predictions."""
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X_test, verbose=0)
    
    if preds.ndim > 1:
        return np.argmax(preds, axis=1)
    else:
        return (preds > 0.5).astype(int)


def save_eval_results(y_true, y_pred, classes, out_prefix):
    """Save classification report and confusion matrix."""
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    # Save text report
    txt_path = out_prefix + "_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix image
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes, annot=False)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(os.path.basename(out_prefix))
    plt.tight_layout()
    img_path = out_prefix + "_cm.png"
    plt.savefig(img_path, dpi=100)
    plt.close()
    
    print(f"  ✓ Report: {txt_path}")
    print(f"  ✓ Matrix: {img_path}")


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():
    print("\n" + "="*70)
    print("MODEL EVALUATION: STATIC + DYNAMIC")
    print("="*70)
    
    # =====================================================
    # STATIC MODELS
    # =====================================================
    
    print("\n[1] STATIC MODELS")
    print("-" * 70)
    
    X_static, y_static, classes_static = load_dataset(DATA_DIR_STATIC)
    
    if X_static is not None and len(X_static) > 0:
        _, X_test_static, _, y_test_static = train_test_split(
            X_static, y_static, test_size=0.2, stratify=y_static, random_state=SEED
        )
        
        le_static = LabelEncoder()
        y_test_static_encoded = le_static.fit_transform(y_test_static)
        classes_static_list = list(le_static.classes_)
        
        static_models, scalers = get_static_models()
        
        if static_models:
            out_dir_static = os.path.join(OUTPUT_DIR, "static")
            os.makedirs(out_dir_static, exist_ok=True)
            
            for model_name, model_path in sorted(static_models.items()):
                try:
                    # Find matching scaler
                    scaler = None
                    for scaler_name, scaler_path in scalers.items():
                        if "static" in scaler_name.lower():
                            scaler = joblib.load(scaler_path)
                            break
                    
                    print(f"\nEvaluating: {model_name}")
                    y_pred = predict_static(model_path, X_test_static, scaler)
                    
                    out_prefix = os.path.join(out_dir_static, model_name.replace(".pkl", ""))
                    save_eval_results(y_test_static_encoded, y_pred, classes_static_list, out_prefix)
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
        else:
            print("No static models found.")
    else:
        print(f"Dataset not found: {DATA_DIR_STATIC}")
    
    # =====================================================
    # DYNAMIC MODELS
    # =====================================================
    
    print("\n[2] DYNAMIC MODELS")
    print("-" * 70)
    
    X_dynamic, y_dynamic, classes_dynamic, seq_shape = prepare_dynamic_dataset(DATA_DIR_DYNAMIC)
    
    if X_dynamic is not None and len(X_dynamic) > 0:
        _, X_test_dynamic, _, y_test_dynamic = train_test_split(
            X_dynamic, y_dynamic, test_size=0.2, stratify=y_dynamic, random_state=SEED
        )
        
        le_dynamic = LabelEncoder()
        y_test_dynamic_encoded = le_dynamic.fit_transform(y_test_dynamic)
        classes_dynamic_list = list(le_dynamic.classes_)
        
        dynamic_models = get_dynamic_models()
        
        if dynamic_models:
            out_dir_dynamic = os.path.join(OUTPUT_DIR, "dynamic")
            os.makedirs(out_dir_dynamic, exist_ok=True)
            
            for model_name, model_path in sorted(dynamic_models.items()):
                try:
                    print(f"\nEvaluating: {model_name}")
                    y_pred = predict_dynamic(model_path, X_test_dynamic)
                    
                    out_prefix = os.path.join(out_dir_dynamic, model_name.replace(".h5", ""))
                    save_eval_results(y_test_dynamic_encoded, y_pred, classes_dynamic_list, out_prefix)
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
        else:
            print("No dynamic models found.")
    else:
        print(f"Dataset not found or empty: {DATA_DIR_DYNAMIC}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
