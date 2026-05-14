import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def summarize_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_static():
    df = pd.read_csv("nsl_landmarks.csv")
    df_clean = df.drop_duplicates()

    X = df_clean.drop("label", axis=1).values
    y = df_clean["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return summarize_metrics(y_test, y_pred)


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


def load_dynamic_dataset(data_dir):
    classes = sorted(os.listdir(data_dir))
    raw_sequences = []
    labels = []

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for file in os.listdir(cls_path):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(cls_path, file))
                raw_sequences.append(seq)
                labels.append(cls)

    if not raw_sequences:
        raise SystemExit(f"No sequences found in {data_dir}")

    target_length = max(seq.shape[0] for seq in raw_sequences if seq.ndim == 2)
    target_dim = max(seq.shape[1] for seq in raw_sequences if seq.ndim == 2)

    X = []
    for seq in raw_sequences:
        normalized = normalize_sequence(seq, target_length, target_dim)
        if normalized is not None:
            X.append(normalized)

    return np.stack(X), np.array(labels)


def evaluate_dynamic():
    data_dir = os.path.join("dataset_word", "landmarks", "final")
    X, y = load_dynamic_dataset(data_dir)

    label_encoder = joblib.load(os.path.join("models", "final", "label_encoder.pkl"))
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    script_dir = os.path.join(os.getcwd(), "scripts")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from improved_model import AttentionLayer

    model = tf.keras.models.load_model(
        os.path.join("models", "final", "best_model.h5"),
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False
    )

    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    return summarize_metrics(y_test, y_pred)


def main():
    static_metrics = evaluate_static()
    dynamic_metrics = evaluate_dynamic()

    results = pd.DataFrame([
        {"Metric": "Accuracy", "STATIC": static_metrics["accuracy"], "DYNAMIC": dynamic_metrics["accuracy"]},
        {"Metric": "Precision", "STATIC": static_metrics["precision"], "DYNAMIC": dynamic_metrics["precision"]},
        {"Metric": "Recall", "STATIC": static_metrics["recall"], "DYNAMIC": dynamic_metrics["recall"]},
        {"Metric": "F1-score", "STATIC": static_metrics["f1"], "DYNAMIC": dynamic_metrics["f1"]}
    ])

    results = results.round(4)

    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "summary_static_vs_dynamic.csv")
    results.to_csv(summary_csv, index=False)

    fig_path = os.path.join(output_dir, "static_vs_dynamic_overall.png")
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(results["Metric"]))
    width = 0.35

    plt.bar(x - width / 2, results["STATIC"], width, label="STATIC")
    plt.bar(x + width / 2, results["DYNAMIC"], width, label="DYNAMIC")
    plt.xticks(x, results["Metric"], rotation=0)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Static vs Dynamic (Overall Macro Metrics)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("\nStatic vs Dynamic (overall macro metrics)\n")
    print(results.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved comparison figure: {fig_path}")


if __name__ == "__main__":
    main()
