

import os
import json
import joblib
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

from improved_model import build_all_models, get_training_config

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "dataset_word/landmarks/final"

MODEL_DIR = "models/final"
OUTPUT_DIR = "outputs/final"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LABEL_SMOOTHING = 0.1

# =========================================================
# AUGMENTATION
# =========================================================

def augment_sequence(seq):

    augmented = []

    # original
    augmented.append(seq)

    # gaussian noise
    noise = seq + np.random.normal(0, 0.01, seq.shape)
    augmented.append(noise)

    # temporal shift
    shift = np.roll(seq, np.random.randint(1, 5), axis=0)
    augmented.append(shift)

    # crop + pad
    crop_len = int(len(seq) * 0.8)

    start = (len(seq) - crop_len) // 2

    cropped = seq[start:start + crop_len]

    pad = np.repeat(
        cropped[-1:],
        len(seq) - crop_len,
        axis=0
    )

    cropped = np.vstack((cropped, pad))

    augmented.append(cropped)

    # feature dropout
    dropout_mask = np.random.binomial(
        1,
        0.9,
        seq.shape
    )

    dropped = seq * dropout_mask

    augmented.append(dropped)

    return augmented


# =========================================================
# MIXUP AUGMENTATION
# =========================================================

def mixup_data(X, y, alpha=0.2):

    indices = np.random.permutation(len(X))

    X2 = X[indices]
    y2 = y[indices]

    lam = np.random.beta(alpha, alpha)

    X_mix = lam * X + (1 - lam) * X2

    return X_mix, y, y2, lam


# =========================================================
# TEMPORAL CONSISTENCY LOSS
# =========================================================

class TemporalConsistencyLoss(tf.keras.losses.Loss):

    def __init__(self,
                 num_classes,
                 label_smoothing=0.1,
                 temporal_weight=0.05):

        super().__init__()

        self.num_classes = num_classes
        self.temporal_weight = temporal_weight

        self.ce = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )

    def call(self, y_true, y_pred):

        y_true = tf.one_hot(
            tf.cast(y_true, tf.int32),
            depth=self.num_classes
        )

        ce_loss = self.ce(y_true, y_pred)

        temporal_loss = tf.reduce_mean(
            tf.square(
                y_pred[1:] - y_pred[:-1]
            )
        )

        return ce_loss + (
            self.temporal_weight * temporal_loss
        )


# =========================================================
# LOAD DATA
# =========================================================

print("\n[1] Loading dataset...")

X = []
y = []


def normalize_sequence(seq, target_length, target_dim):

    seq = np.asarray(seq)

    if seq.ndim != 2:
        return None

    # Fix time dimension first
    if seq.shape[0] < target_length:
        pad_rows = np.zeros((target_length - seq.shape[0], seq.shape[1]), dtype=seq.dtype)
        seq = np.vstack([seq, pad_rows])
    elif seq.shape[0] > target_length:
        seq = seq[:target_length]

    # Then fix feature dimension
    if seq.shape[1] < target_dim:
        pad_cols = np.zeros((seq.shape[0], target_dim - seq.shape[1]), dtype=seq.dtype)
        seq = np.hstack([seq, pad_cols])
    elif seq.shape[1] > target_dim:
        seq = seq[:, :target_dim]

    return seq

classes = sorted(os.listdir(DATA_DIR))

raw_sequences = []

for cls in classes:

    path = os.path.join(DATA_DIR, cls)

    if not os.path.isdir(path):
        continue

    for file in os.listdir(path):

        if file.endswith(".npy"):

            seq = np.load(
                os.path.join(path, file)
            )

            raw_sequences.append(seq)
            y.append(cls)

if not raw_sequences:
    raise SystemExit(f"No .npy landmark sequences found in {DATA_DIR}")

target_length = max(seq.shape[0] for seq in raw_sequences if seq.ndim == 2)
target_dim = max(seq.shape[1] for seq in raw_sequences if seq.ndim == 2)

for seq in raw_sequences:
    normalized = normalize_sequence(seq, target_length, target_dim)
    if normalized is not None:
        X.append(normalized)

X = np.stack(X)
y = np.array(y)

print(f"Loaded {len(X)} sequences with shape {X.shape[1:]}")

print(f"Original samples: {len(X)}")
print(f"Total classes: {len(classes)}")

# =========================================================
# LABEL ENCODING
# =========================================================

le = LabelEncoder()

y_encoded = le.fit_transform(y)

joblib.dump(
    le,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

# =========================================================
# TRAIN / VAL / TEST SPLIT
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full
)

print("\nDataset Split:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =========================================================
# AUGMENT TRAIN ONLY
# =========================================================

print("\n[2] Applying augmentation...")

X_train_aug = []
y_train_aug = []

for seq, label in zip(X_train, y_train):

    augmented = augment_sequence(seq)

    for a in augmented:

        X_train_aug.append(a)
        y_train_aug.append(label)

X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)

print(f"Augmented train samples: {len(X_train)}")

# =========================================================
# AFTER AUGMENTATION REPORT
# =========================================================

print("\n==================================================")
print("AFTER AUGMENTATION REPORT")
print("==================================================")

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

print("\nClass Distribution (TRAIN):")

unique_classes, counts = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(unique_classes, counts):

    class_name = le.inverse_transform([cls])[0]

    print(f"{class_name}: {count}")

# =========================================================
# CLASS WEIGHTS
# =========================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    enumerate(class_weights)
)

# =========================================================
# MODEL SETUP
# =========================================================

config = get_training_config(len(classes))

input_shape = (
    X.shape[1],
    X.shape[2]
)

models_dict = build_all_models(
    input_shape,
    len(classes)
)

results = {}

# =========================================================
# TRAINING LOOP
# =========================================================

for model_name, model_fn in models_dict.items():

    print("\n" + "="*70)
    print(f"TRAINING MODEL: {model_name}")
    print("="*70)

    model = model_fn

    # cosine learning rate
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config["learning_rate"],
        decay_steps=1000
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    loss_fn = TemporalConsistencyLoss(
        num_classes=len(classes),
        label_smoothing=LABEL_SMOOTHING,
        temporal_weight=0.05
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        # Removed ReduceLROnPlateau because optimizer uses a LearningRateSchedule
        # (CosineDecay). ReduceLROnPlateau attempts to set optimizer.learning_rate
        # which is not allowed when a schedule object was provided.

        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                MODEL_DIR,
                f"{model_name}_best.h5"
            ),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # =====================================================
    # MIXUP
    # =====================================================

    X_mix, y_a, y_b, lam = mixup_data(
        X_train,
        y_train
    )

    history = model.fit(
        X_mix,
        y_a,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    val_pred = np.argmax(
        model.predict(X_val),
        axis=1
    )

    val_acc = accuracy_score(
        y_val,
        val_pred
    )

    results[model_name] = val_acc

    print(f"\nValidation Accuracy: {val_acc:.4f}")

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(
        y_val,
        val_pred,
        target_names=le.classes_
    )

    print(report)

    with open(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_report.txt"
        ),
        "w"
    ) as f:

        f.write(report)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_val,
        val_pred
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_cm.png"
        )
    )

    plt.close()

# =========================================================
# BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=results.get
)

print("\n🏆 BEST MODEL:", best_model_name)

# =========================================================
# FINAL MODEL TRAINING
# =========================================================

print("\n[FINAL TRAINING]")

X_final = np.concatenate([
    X_train,
    X_val
])

y_final = np.concatenate([
    y_train,
    y_val
])

final_model = build_all_models(
    input_shape,
    len(classes)
)[best_model_name]

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"]
    ),
    loss=TemporalConsistencyLoss(
        num_classes=len(classes),
        label_smoothing=LABEL_SMOOTHING,
        temporal_weight=0.05
    ),
    metrics=["accuracy"]
)

final_model.fit(
    X_final,
    y_final,
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    class_weight=class_weights,
    verbose=1
)

# =========================================================
# TEST EVALUATION
# =========================================================

test_pred = np.argmax(
    final_model.predict(X_test),
    axis=1
)

test_acc = accuracy_score(
    y_test,
    test_pred
)

print("\n🎯 FINAL TEST ACCURACY:", test_acc)

final_report = classification_report(
    y_test,
    test_pred,
    target_names=le.classes_
)

print(final_report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "final_test_report.txt"
    ),
    "w"
) as f:

    f.write(final_report)

# =========================================================
# FINAL TEST CONFUSION MATRIX
# =========================================================

cm_test = confusion_matrix(
    y_test,
    test_pred
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm_test,
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("FINAL TEST CONFUSION MATRIX")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_test_cm.png"
    )
)

plt.close()

# =========================================================
# SAVE FINAL MODEL
# =========================================================

final_model.save(
    os.path.join(
        MODEL_DIR,
        "best_model.h5"
    )
)

meta = {

    "best_model": best_model_name,

    "validation_accuracy":
        float(results[best_model_name]),

    "test_accuracy":
        float(test_acc),

    "classes":
        list(le.classes_),

    "input_shape":
        input_shape
}

with open(
    os.path.join(
        MODEL_DIR,
        "meta.json"
    ),
    "w"
) as f:

    json.dump(meta, f, indent=2)

print("\n✅ TRAINING COMPLETE")