"""Comprehensive Model Evaluation: All 8 Sections (4.8.1-4.8.8)

Usage: run from repo root:
    python scripts/advanced_evaluation.py

Covers all sections of 4.8 Model Evaluation and Error Analysis:
    4.8.1 Experimental Evaluation Framework
    4.8.2 Accuracy, Precision, Recall, and F1-Score Analysis
    4.8.3 Confusion Matrix Analysis
    4.8.4 ROC Curve and AUC Evaluation
    4.8.5 Class-Wise Performance Analysis (enhanced)
    4.8.6 Misclassification Pattern Investigation
    4.8.7 Occlusion and Landmark Failure Analysis
    4.8.8 Real-Time Prediction Stability Analysis
"""

import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_curve
)

# Add scripts directory to path for custom imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import custom objects from improved_model
try:
    from improved_model import AttentionLayer, LabelSmoothingLoss, TemporalConsistencyLoss
except ImportError:
    print("Warning: Could not import custom objects from improved_model")
    AttentionLayer = None
    LabelSmoothingLoss = None
    TemporalConsistencyLoss = None


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.getcwd()
if not os.path.exists(os.path.join(BASE_DIR, "dataset_word")):
    parent = os.path.abspath(os.path.join(BASE_DIR, ".."))
    if os.path.exists(os.path.join(parent, "dataset_word")):
        BASE_DIR = parent

DATA_DIR_DYNAMIC = os.path.join(BASE_DIR, "dataset_word", "landmarks", "final")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "advanced_eval")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DATASET LOADING
# =========================================================

def load_dataset(data_dir):
    """Load .npy files from directory structure (returns raw sequences)."""
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
    
    # Return as lists (not arrays) to preserve variable lengths
    # Normalization happens in prepare_dataset()
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


def prepare_dataset(data_dir):
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
# 4.8.1 EXPERIMENTAL EVALUATION FRAMEWORK
# =========================================================

def setup_evaluation_framework(model_path, X_test, y_test_encoded, classes, output_dir):
    """
    Setup experimental evaluation framework.
    
    Loads model, prepares test data, generates predictions.
    Serves as foundation for all subsequent evaluations.
    """
    print(f"  [4.8.1] Setting up experimental framework...")
    
    # Load model with custom objects - simpler approach for evaluation only
    # We don't need to recompile the model since we're only doing inference
    custom_objects = {}
    if AttentionLayer is not None:
        custom_objects['AttentionLayer'] = AttentionLayer
    if LabelSmoothingLoss is not None:
        custom_objects['LabelSmoothingLoss'] = LabelSmoothingLoss
    if TemporalConsistencyLoss is not None:
        custom_objects['TemporalConsistencyLoss'] = TemporalConsistencyLoss
    
    # Load model with custom_object_scope to ensure custom objects are available
    # This MUST wrap the entire load_model call
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path, compile=False)
    
    # Generate predictions
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Framework metadata
    framework_meta = {
        "model_path": model_path,
        "model_name": os.path.basename(model_path),
        "test_set_size": len(X_test),
        "num_classes": len(classes),
        "classes": list(classes),
        "input_shape": str(X_test.shape),
        "output_shape": str(y_proba.shape)
    }
    
    framework_json = os.path.join(output_dir, "framework_metadata.json")
    with open(framework_json, "w") as f:
        json.dump(framework_meta, f, indent=2)
    
    print(f"    [OK] Framework setup complete")
    print(f"    [OK] Test set: {len(X_test)} samples, {len(classes)} classes")
    print(f"    [OK] Metadata: {framework_json}")
    
    return model, y_proba, y_pred


# =========================================================
# 4.8.2 ACCURACY, PRECISION, RECALL, F1-SCORE ANALYSIS
# =========================================================

def compute_basic_metrics(y_true, y_pred, classes, output_dir):
    """
    Compute and analyze basic classification metrics:
    Accuracy, Precision, Recall, F1-Score
    """
    print(f"  [4.8.2] Computing basic classification metrics...")
    
    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Per-class and weighted metrics
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    
    # Extract metrics
    metrics_data = {
        "overall_accuracy": float(acc),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"])
    }
    
    # Save classification report
    report_txt = os.path.join(output_dir, "classification_report.txt")
    with open(report_txt, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    # Save JSON metrics
    metrics_json = os.path.join(output_dir, "basic_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_list = ["Overall\nAccuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1-Score"]
    values = [metrics_data["overall_accuracy"], metrics_data["macro_precision"],
              metrics_data["macro_recall"], metrics_data["macro_f1"]]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    bars = ax.bar(metrics_list, values, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylim([0, 1.0])
    ax.set_ylabel("Score")
    ax.set_title("Basic Classification Metrics Overview")
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    metrics_plot = os.path.join(output_dir, "basic_metrics.png")
    plt.savefig(metrics_plot, dpi=150)
    plt.close()
    
    print(f"    [OK] Classification report: {report_txt}")
    print(f"    [OK] Metrics JSON: {metrics_json}")
    print(f"    [OK] Metrics visualization: {metrics_plot}")
    print(f"    [OK] Overall Accuracy: {metrics_data['overall_accuracy']:.4f}")
    print(f"    [OK] Weighted F1-Score: {metrics_data['weighted_f1']:.4f}")
    
    return metrics_data


# =========================================================
# 4.8.3 CONFUSION MATRIX ANALYSIS
# =========================================================

def confusion_matrix_analysis(y_true, y_pred, classes, output_dir):
    """
    Generate and analyze confusion matrix with multiple visualizations.
    """
    print(f"  [4.8.3] Analyzing confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Raw confusion matrix (counts)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes, cbar_kws={"label": "Count"})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (Raw Counts)")
    plt.tight_layout()
    
    cm_raw_path = os.path.join(output_dir, "confusion_matrix_raw.png")
    plt.savefig(cm_raw_path, dpi=150)
    plt.close()
    
    # Normalized confusion matrix (row-wise percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar_kws={"label": "%"})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (Normalized by True Class %)")
    plt.tight_layout()
    
    cm_norm_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_norm_path, dpi=150)
    plt.close()
    
    # Compute per-class accuracy (diagonal elements)
    per_class_acc = np.diag(cm) / cm.sum(axis=1)
    
    cm_stats = {
        "total_samples": int(cm.sum()),
        "correct_predictions": int(np.trace(cm)),
        "per_class_recall": {classes[i]: float(per_class_acc[i]) for i in range(len(classes))}
    }
    
    # Save confusion matrix data
    cm_json = os.path.join(output_dir, "confusion_matrix_stats.json")
    with open(cm_json, "w") as f:
        json.dump(cm_stats, f, indent=2)
    
    # Save as CSV for reference
    cm_csv = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(cm_csv)
    
    print(f"    [OK] Confusion matrix (raw counts): {cm_raw_path}")
    print(f"    [OK] Confusion matrix (normalized): {cm_norm_path}")
    print(f"    [OK] Confusion matrix stats: {cm_json}")
    print(f"    [OK] Confusion matrix CSV: {cm_csv}")
    print(f"    [OK] Total correct: {cm_stats['correct_predictions']}/{cm_stats['total_samples']}")
    
    return cm, cm_stats


# =========================================================
# 4.8.4 ROC CURVE AND AUC EVALUATION
# =========================================================

def compute_roc_auc(model_path, X_test, y_test_encoded, classes, output_dir):
    """Compute and plot ROC curves and AUC for multi-class classification."""
    print(f"  Computing ROC/AUC...")
    
    custom_objects = {}
    if AttentionLayer is not None:
        custom_objects['AttentionLayer'] = AttentionLayer
    if LabelSmoothingLoss is not None:
        custom_objects['LabelSmoothingLoss'] = LabelSmoothingLoss
    if TemporalConsistencyLoss is not None:
        custom_objects['TemporalConsistencyLoss'] = TemporalConsistencyLoss
    
    # Try loading with compile=False first to avoid compilation issues
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
    
    y_proba = model.predict(X_test, verbose=0)
    
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Per-class ROC
    for i in range(n_classes):
        y_binary = (y_test_encoded == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_binary, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    
    # Plot ROC curves
    plt.figure(figsize=(14, 10))
    colors = cycle(plt.cm.tab20(np.linspace(0, 1, n_classes)))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")
    
    plt.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro-average (AUC = {roc_auc['macro']:.3f})")
    plt.plot([0, 1], [0, 1], "k-", lw=0.5, label="Random Chance")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    # Save AUC scores
    auc_summary = {"macro_auc": float(roc_auc["macro"])}
    for i, cls in enumerate(classes):
        auc_summary[cls] = float(roc_auc[i])
    
    auc_json = os.path.join(output_dir, "auc_scores.json")
    with open(auc_json, "w") as f:
        json.dump(auc_summary, f, indent=2)
    
    print(f"    [OK] ROC curves: {roc_path}")
    print(f"    [OK] AUC scores: {auc_json} (Macro AUC: {roc_auc['macro']:.4f})")


# =========================================================
# 4.8.5 CLASS-WISE PERFORMANCE ANALYSIS
# =========================================================

def class_wise_performance(y_true, y_pred, classes, output_dir):
    """Detailed per-class performance breakdown."""
    print(f"  Computing class-wise metrics...")
    
    report_dict = classification_report(y_true, y_pred, target_names=classes,
                                        output_dict=True, zero_division=0)
    
    class_metrics = {}
    for cls in classes:
        if cls in report_dict:
            class_metrics[cls] = {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1-score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
    
    # Visualize per-class metrics
    df_metrics = pd.DataFrame(class_metrics).T
    df_metrics_sorted = df_metrics.sort_values("f1-score", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_metrics_sorted[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax)
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.title("Per-Class Performance Metrics")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    cls_perf_path = os.path.join(output_dir, "class_performance.png")
    plt.savefig(cls_perf_path, dpi=150)
    plt.close()
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "class_metrics.csv")
    df_metrics.to_csv(csv_path)
    
    print(f"    [OK] Class performance plot: {cls_perf_path}")
    print(f"    [OK] Class metrics CSV: {csv_path}")
    
    return class_metrics


# =========================================================
# 4.8.6 MISCLASSIFICATION PATTERN INVESTIGATION
# =========================================================

def misclassification_analysis(y_true, y_pred, classes, output_dir):
    """Analyze and visualize misclassification patterns."""
    print(f"  Analyzing misclassification patterns...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix to show percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar_kws={"label": "%"})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalized Confusion Matrix (%)")
    plt.tight_layout()
    
    cm_norm_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_norm_path, dpi=150)
    plt.close()
    
    # Top misclassification pairs
    misclass_pairs = []
    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            if i != j:
                count = cm[i, j]
                if count > 0:
                    misclass_pairs.append({
                        "true_class": cls_true,
                        "predicted_class": cls_pred,
                        "count": count,
                        "percentage": (count / cm[i].sum()) * 100
                    })
    
    misclass_df = pd.DataFrame(misclass_pairs).sort_values("count", ascending=False)
    
    # Top 20 misclassifications
    top_misclass_path = os.path.join(output_dir, "top_misclassifications.csv")
    misclass_df.head(20).to_csv(top_misclass_path, index=False)
    
    # Visualize top misclassifications
    if len(misclass_df) > 0:
        top_20 = misclass_df.head(20)
        top_20["pair"] = top_20["true_class"] + " → " + top_20["predicted_class"]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_20)), top_20["count"])
        plt.yticks(range(len(top_20)), top_20["pair"], fontsize=9)
        plt.xlabel("Misclassification Count")
        plt.title("Top 20 Misclassification Pairs")
        plt.tight_layout()
        
        misclass_viz_path = os.path.join(output_dir, "top_misclassifications.png")
        plt.savefig(misclass_viz_path, dpi=150)
        plt.close()
        
        print(f"    [OK] Confusion matrix (normalized): {cm_norm_path}")
        print(f"    [OK] Top misclassifications: {top_misclass_path}")
        print(f"    [OK] Misclassification plot: {misclass_viz_path}")


# =========================================================
# 4.8.7 OCCLUSION AND LANDMARK FAILURE ANALYSIS
# =========================================================

def occlusion_robustness_test(model_path, X_test, y_test_encoded, classes, output_dir):
    """Test model robustness under landmark dropout (occlusion)."""
    print(f"  Testing occlusion robustness...")
    
    custom_objects = {}
    if AttentionLayer is not None:
        custom_objects['AttentionLayer'] = AttentionLayer
    if LabelSmoothingLoss is not None:
        custom_objects['LabelSmoothingLoss'] = LabelSmoothingLoss
    if TemporalConsistencyLoss is not None:
        custom_objects['TemporalConsistencyLoss'] = TemporalConsistencyLoss
    
    # Try loading with compile=False first to avoid compilation issues
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
    
    # Original accuracy (no occlusion)
    y_pred_orig = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc_orig = accuracy_score(y_test_encoded, y_pred_orig)
    
    # Test with increasing occlusion levels
    occlusion_levels = [10, 20, 30, 40, 50]  # percentage of features zeroed
    accuracies = [acc_orig]
    
    for occ_pct in occlusion_levels:
        X_occluded = X_test.copy()
        n_features = X_occluded.shape[2]
        n_drop = int(n_features * occ_pct / 100)
        
        # Randomly zero out features
        for sample in X_occluded:
            drop_idx = np.random.choice(n_features, n_drop, replace=False)
            sample[:, drop_idx] = 0
        
        y_pred_occ = np.argmax(model.predict(X_occluded, verbose=0), axis=1)
        acc_occ = accuracy_score(y_test_encoded, y_pred_occ)
        accuracies.append(acc_occ)
    
    # Plot robustness
    occ_labels = ["0% (Original)"] + [f"{p}%" for p in occlusion_levels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(occ_labels, accuracies, marker="o", linewidth=2, markersize=8)
    plt.ylabel("Accuracy")
    plt.xlabel("Occlusion Level (50% of Features Zeroed)")
    plt.title("Model Robustness Under Landmark Occlusion")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    occlusion_path = os.path.join(output_dir, "occlusion_robustness.png")
    plt.savefig(occlusion_path, dpi=150)
    plt.close()
    
    # Save results
    occlusion_data = {
        "occlusion_levels": occ_labels,
        "accuracies": [float(a) for a in accuracies],
        "degradation_pct_50": float((acc_orig - accuracies[-1]) / acc_orig * 100)
    }
    
    occ_json_path = os.path.join(output_dir, "occlusion_results.json")
    with open(occ_json_path, "w") as f:
        json.dump(occlusion_data, f, indent=2)
    
    print(f"    [OK] Occlusion robustness plot: {occlusion_path}")
    print(f"    [OK] Robustness data: {occ_json_path}")
    print(f"    [OK] Accuracy degradation at 50% occlusion: {occlusion_data['degradation_pct_50']:.2f}%")


# =========================================================
# 4.8.8 REAL-TIME PREDICTION STABILITY ANALYSIS
# =========================================================

def stability_analysis(model_path, X_test, y_test_encoded, classes, output_dir):
    """Analyze prediction stability: frame-by-frame consistency and latency."""
    print(f"  Analyzing prediction stability...")
    
    custom_objects = {}
    if AttentionLayer is not None:
        custom_objects['AttentionLayer'] = AttentionLayer
    if LabelSmoothingLoss is not None:
        custom_objects['LabelSmoothingLoss'] = LabelSmoothingLoss
    if TemporalConsistencyLoss is not None:
        custom_objects['TemporalConsistencyLoss'] = TemporalConsistencyLoss
    
    # Try loading with compile=False first to avoid compilation issues
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
    
    # Latency measurements: warm-up + 10 predictions
    import time
    latencies = []
    
    model.predict(X_test[:1], verbose=0)  # Warm-up
    
    for i in range(min(100, len(X_test))):
        start = time.time()
        _ = model.predict(X_test[i:i+1], verbose=0)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    latencies = np.array(latencies)
    
    # Prediction consistency: multiple runs on same sample
    consistency_scores = []
    n_samples = min(50, len(X_test))
    n_runs = 5
    
    for i in range(n_samples):
        predictions = []
        for _ in range(n_runs):
            pred = np.argmax(model.predict(X_test[i:i+1], verbose=0))
            predictions.append(pred)
        
        # Consistency = fraction of runs with same prediction
        consistency = (predictions.count(predictions[0]) / n_runs)
        consistency_scores.append(consistency)
    
    consistency_scores = np.array(consistency_scores)
    
    # Plot latency distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(latencies, bins=20, edgecolor="black", alpha=0.7)
    ax1.axvline(np.mean(latencies), color="r", linestyle="--", linewidth=2, label=f"Mean: {np.mean(latencies):.2f}ms")
    ax1.axvline(np.median(latencies), color="g", linestyle="--", linewidth=2, label=f"Median: {np.median(latencies):.2f}ms")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Prediction Latency Distribution")
    ax1.legend()
    
    ax2.hist(consistency_scores, bins=20, edgecolor="black", alpha=0.7, range=(0, 1))
    ax2.set_xlabel("Consistency Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Prediction Consistency (5 runs per sample)")
    
    plt.tight_layout()
    stability_path = os.path.join(output_dir, "stability_analysis.png")
    plt.savefig(stability_path, dpi=150)
    plt.close()
    
    # Save stability metrics
    stability_metrics = {
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        },
        "consistency": {
            "mean": float(np.mean(consistency_scores)),
            "std": float(np.std(consistency_scores)),
            "min": float(np.min(consistency_scores)),
            "max": float(np.max(consistency_scores))
        }
    }
    
    stability_json_path = os.path.join(output_dir, "stability_metrics.json")
    with open(stability_json_path, "w") as f:
        json.dump(stability_metrics, f, indent=2)
    
    print(f"    [OK] Stability analysis plot: {stability_path}")
    print(f"    [OK] Stability metrics: {stability_json_path}")
    print(f"    [OK] Mean latency: {stability_metrics['latency_ms']['mean']:.2f}ms")
    print(f"    [OK] Mean consistency: {stability_metrics['consistency']['mean']:.4f}")


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION (4.8.1-4.8.8)")
    print("All Sections of Model Evaluation and Error Analysis")
    print("="*70)
    
    # Prepare dataset
    print("\n[SETUP] Loading and preparing dataset...")
    X, y, classes_list, seq_shape = prepare_dataset(DATA_DIR_DYNAMIC)
    
    if X is None or len(X) == 0:
        print(f"[ERROR] Dataset not found or empty: {DATA_DIR_DYNAMIC}")
        return
    
    print(f"[OK] Loaded {len(X)} sequences, {len(classes_list)} classes")
    print(f"  Shape: {X.shape}, Sequence length: {seq_shape}")
    
    # Train/test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    classes_list = list(le.classes_)
    
    # Find and evaluate dynamic models
    print("\n[MODELS] Searching for trained models...")
    h5_files = glob.glob(os.path.join(MODEL_DIR, "**", "*.h5"), recursive=True)
    dynamic_models = [h for h in h5_files if "best_model.h5" in h or "dynamic" in h]
    
    if not dynamic_models:
        print("✗ No dynamic models found in models/")
        return
    
    print(f"[OK] Found {len(dynamic_models)} model(s)")
    
    # Evaluate each model
    for model_path in dynamic_models:
        model_name = os.path.basename(model_path).replace(".h5", "")
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Load model and get predictions
            print(f"\n[4.8.1] Experimental Evaluation Framework")
            model, y_proba, y_pred = setup_evaluation_framework(model_path, X_test, y_test_encoded, 
                                                                 classes_list, model_dir)
            
            print(f"\n[4.8.2] Accuracy, Precision, Recall, F1-Score Analysis")
            metrics_data = compute_basic_metrics(y_test_encoded, y_pred, classes_list, model_dir)
            
            print(f"\n[4.8.3] Confusion Matrix Analysis")
            cm, cm_stats = confusion_matrix_analysis(y_test_encoded, y_pred, classes_list, model_dir)
            
            print(f"\n[4.8.4] ROC Curve and AUC Evaluation")
            compute_roc_auc(model_path, X_test, y_test_encoded, classes_list, model_dir)
            
            print(f"\n[4.8.5] Class-Wise Performance Analysis")
            class_wise_performance(y_test_encoded, y_pred, classes_list, model_dir)
            
            print(f"\n[4.8.6] Misclassification Pattern Investigation")
            misclassification_analysis(y_test_encoded, y_pred, classes_list, model_dir)
            
            print(f"\n[4.8.7] Occlusion and Landmark Failure Analysis")
            occlusion_robustness_test(model_path, X_test, y_test_encoded, classes_list, model_dir)
            
            print(f"\n[4.8.8] Real-Time Prediction Stability Analysis")
            stability_analysis(model_path, X_test, y_test_encoded, classes_list, model_dir)
            
            print(f"\n[SUCCESS] All analyses complete for {model_name}")
            
        except Exception as e:
            print(f"\n[ERROR] Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

