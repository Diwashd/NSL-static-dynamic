"""
ABLATION STUDY - Dynamic Nepali Sign Language Recognition
Systematic evaluation for thesis
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from improved_model import (
    build_all_models, 
    TemporalConsistencyLoss, 
    LabelSmoothingLoss
)

# ========================= CONFIG =========================
DATA_DIR = "dataset_word/landmarks/final"
ABLATION_DIR = "ablation_results"
MODEL_DIR = "ablation_models"
os.makedirs(ABLATION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===================== FEATURE MODES =====================
def normalize_sequence(seq, target_length, target_dim):
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


def load_data(feature_mode="full", target_length=40):
    """Load data with different feature configurations"""
    print(f"Loading data with mode: {feature_mode}")
    
    # Resolve DATA_DIR robustly: allow running from notebook, script, or different cwd.
    data_dir = DATA_DIR
    if not os.path.isdir(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        project_root = os.path.dirname(script_dir)
        candidates = [
            os.path.join(script_dir, DATA_DIR),
            os.path.join(project_root, DATA_DIR),
            os.path.abspath(DATA_DIR),
        ]

        for candidate in candidates:
            if os.path.isdir(candidate):
                data_dir = candidate
                break
        else:
            tried = "', '".join(candidates)
            raise SystemExit(
                f"DATA_DIR '{DATA_DIR}' not found. Tried '{tried}'. Please set DATA_DIR correctly or run from project root."
            )

    raw_sequences = []
    y = []
    
    classes = sorted(os.listdir(data_dir))
    for cls in classes:
        path = os.path.join(data_dir, cls)
        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(path, file))
                if seq.ndim != 2:
                    continue
                
                # Feature selection
                if feature_mode == "landmarks_only":
                    # Hand (21*3) + Pose (12*3) = 99 dims
                    seq = seq[:, :99]
                elif feature_mode == "landmarks_vel":
                    # Landmarks + Velocity
                    seq = seq[:, :198]   # 99*2
                elif feature_mode == "no_geometric":
                    # All except last 3 features
                    seq = seq[:, :-3]
                elif feature_mode == "no_bones":
                    # Remove bone features (after velocity+accel)
                    seq = np.concatenate([seq[:, :198], seq[:, -3:]], axis=1)
                # "full" = use all 363 dimensions
                
                raw_sequences.append(seq)
                y.append(cls)
    
    if not raw_sequences:
        raise SystemExit(f"No .npy landmark sequences found in {DATA_DIR}")

    target_dim = max(seq.shape[1] for seq in raw_sequences if seq.ndim == 2)

    X = []
    for seq in raw_sequences:
        normalized = normalize_sequence(seq, target_length, target_dim)
        if normalized is not None:
            X.append(normalized)

    X = np.stack(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} sequences, shape: {X.shape}")
    return X, y, classes


# ===================== MAIN ABLATION FUNCTION =====================
def run_ablation(experiment_name, 
                 feature_mode="full",
                 model_type="BiGRU_Attention",
                 use_temporal_loss=True,
                 seq_length=40,
                 epochs=60,
                 note=""):
    
    print(f"\n{'='*90}")
    print(f"STARTING ABLATION: {experiment_name}")
    print(f"Feature: {feature_mode} | Model: {model_type} | Temporal Loss: {use_temporal_loss} | Seq Len: {seq_length}")
    print(f"{'='*90}")
    
    # Load data
    X, y, class_names = load_data(feature_mode, target_length=seq_length)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/Val/Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED)
    
    input_shape = (seq_length, X.shape[2])
    
    # Build Model
    models_dict = build_all_models(input_shape, num_classes=len(class_names))
    model = models_dict[model_type]
    
    # Loss
    if use_temporal_loss:
        loss_fn = TemporalConsistencyLoss(
            num_classes=len(class_names), 
            label_smoothing=0.1, 
            temporal_weight=0.05
        )
    else:
        loss_fn = LabelSmoothingLoss(num_classes=len(class_names), smoothing=0.1)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{experiment_name}_best.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=0)
    ]
    
    # Training
    start = datetime.now()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    train_time = (datetime.now() - start).total_seconds()
    
    # Evaluation
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Save result
    result = {
        "experiment": experiment_name,
        "feature_mode": feature_mode,
        "model_type": model_type,
        "use_temporal_loss": use_temporal_loss,
        "seq_length": seq_length,
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "training_time_sec": train_time,
        "note": note,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    # Save to CSV
    csv_path = os.path.join(ABLATION_DIR, "ablation_results.csv")
    pd.DataFrame([result]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    
    print(f"✅ FINISHED → Val: {val_acc:.4f} | Test: {test_acc:.4f} | Time: {train_time:.1f}s\n")
    return result


# ========================= RUN ABLATIONS =========================
if __name__ == "__main__":
    
    experiments = [
        # 1. Feature Ablation
        ("01_Full_Features",          "full",             "BiGRU_Attention", True, 40),
        ("02_Landmarks_Only",         "landmarks_only",   "BiGRU_Attention", True, 40),
        ("03_Landmarks_Vel",          "landmarks_vel",    "BiGRU_Attention", True, 40),
        ("04_No_Geometric",           "no_geometric",     "BiGRU_Attention", True, 40),
        ("05_No_Bones",               "no_bones",         "BiGRU_Attention", True, 40),
        
        # 2. Model Architecture Ablation
        ("06_GRU_Baseline",           "full",             "GRU",             True, 40),
        ("07_BiGRU",                  "full",             "BiGRU",           True, 40),
        ("08_BiGRU_Attention",        "full",             "BiGRU_Attention", True, 40),
        
        # 3. Loss Ablation
        ("09_No_Temporal_Loss",       "full",             "BiGRU_Attention", False, 40),
        
        # 4. Sequence Length
        ("10_SeqLen_20",              "full",             "BiGRU_Attention", True, 20),
        ("11_SeqLen_30",              "full",             "BiGRU_Attention", True, 30),
        ("12_SeqLen_60",              "full",             "BiGRU_Attention", True, 60),
    ]
    
    print("🚀 Starting Ablation Study...\n")
    
    for exp in experiments:
        run_ablation(
            experiment_name=exp[0],
            feature_mode=exp[1],
            model_type=exp[2],
            use_temporal_loss=exp[3],
            seq_length=exp[4],
            epochs=65,           # Reduced for faster ablation
            note="Ablation study"
        )
    
    print("\n🎉 ALL ABLATION EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {ABLATION_DIR}/ablation_results.csv")