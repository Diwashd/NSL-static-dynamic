 # 4.10 TRAINING ENVIRONMENT AND SYSTEM CONFIGURATION

## 4.10.1 Hardware Specifications

### Processor and Architecture
- **Processor**: Intel Core (6th Gen or equivalent)
- **CPU Cores**: 8 cores available
- **Architecture**: AMD64 (x86-64)
- **CPU Model**: Intel64 Family 6 Model 158 Stepping 13, GenuineIntel

### Memory Configuration
- **Total RAM**: 11.85 GB
- **Available RAM**: 0.70 GB (at system information collection time)
- **RAM Type**: System physical memory (allocated for OS, training, and inference)

### Storage
- **Dataset Storage**: 29.99 MB (22 gesture classes, ~276 sequences)
  - Location: `dataset_word/landmarks/final/`
  - Contains: 22 subdirectories (one per gesture class)
  - Format: NumPy .npy files (3D sequences)

- **Model Storage**: 13.38 MB total
  - Final Models: 8.31 MB (best_model.h5, GRU_best.h5, BiGRU_best.h5, BiGRU_Attention_best.h5)
  - Dynamic Models: 5.07 MB (nsl_gru_dynamic_model.h5, label encoders)
  - Serialization Format: HDF5 (.h5) for Keras models, pickle (.pkl) for scikit-learn models

- **Output Storage**: 1.14 MB
  - Training outputs: 0.45 MB (reports, confusion matrices, metrics)
  - Evaluation outputs: 0.69 MB (advanced evaluation results per model)

### No Dedicated GPU
- **GPU Status**: Not available on native Windows for TensorFlow >= 2.11
- **Workaround**: TensorFlow-DirectML plugin recommended for GPU acceleration on Windows
- **Impact**: Training performed entirely on CPU

---

## 4.10.2 Software Environment

### Operating System
- **Platform**: Windows 10
- **OS Version**: 10.0.26200 SP0
- **Architecture**: 64-bit (AMD64)

### Python Environment
- **Python Version**: 3.11.8
- **Virtual Environment**: Isolated venv at `./venv/`
- **Python Executable**: `D:\Islington college\masters\nsl-landmark-thesis\venv\Scripts\python.exe`
- **Working Directory**: `D:\Islington college\masters\nsl-landmark-thesis`

### Development Workflow
- **Primary IDE**: Visual Studio Code
- **Execution Modes**:
  1. Jupyter Notebooks: `scripts/dynamic_training.ipynb` for interactive training
  2. Python Scripts: `scripts/advanced_evaluation.py` for batch evaluation
  3. Streamlit App: `app.py` for real-time inference demonstration
  4. Command Line: Direct script execution via Python interpreter

### Dependency Management
- **Package Manager**: pip
- **Requirements Specification**: Virtual environment with pinned versions
- **Reproducibility**: Seed=42 for all random operations (NumPy, TensorFlow, random module)

---

## 4.10.3 Python Libraries and Frameworks

### Core Machine Learning Stack
| Library | Version | Purpose |
|---------|---------|---------|
| **TensorFlow** | 2.21.0 | Deep learning framework, Keras API for model building |
| **scikit-learn** | 1.8.0 | Classification models (static detection), metrics computation |
| **NumPy** | 2.4.3 | Numerical operations, array manipulation, augmentation |
| **Pandas** | 2.3.3 | Data loading, CSV handling, metrics aggregation |

### Computer Vision and Pose Estimation
| Library | Version | Purpose |
|---------|---------|---------|
| **MediaPipe** | 0.10.33 | Hand landmarker, pose estimation, face detection |
| **OpenCV (cv2)** | 4.13.0 | Video capture, frame processing, visualization |

### Utilities and Serialization
| Library | Version | Purpose |
|---------|---------|---------|
| **joblib** | 1.5.3 | Model serialization (static classifier), efficient I/O |
| **Streamlit** | Latest (runtime) | Web UI framework for real-time inference demo |
| **Matplotlib** | (implicit) | Visualization of metrics and confusion matrices |
| **Seaborn** | (implicit) | Statistical data visualization |
| **psutil** | (runtime) | System monitoring (optional) |

### Key Framework Components

#### TensorFlow/Keras Configuration
```python
# Random seed for reproducibility
tf.random.set_seed(SEED=42)

# Memory strategy (CPU-based)
# GPU memory growth would be set if GPU available:
# tf.config.experimental.set_memory_growth(gpu, True)

# Optimization: CosineDecay learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=total_train_steps
)

# Custom objects required for model loading:
# - AttentionLayer (self-attention mechanism)
# - LabelSmoothingLoss (CE loss + 0.1 smoothing)
# - TemporalConsistencyLoss (CE + temporal penalty)
```

#### Scikit-learn Configuration
```python
# Static classifier pipeline
static_model_path: "notebooks/nsl_gesture_classifier.pkl"
confidence_threshold: 0.50
features_per_frame: 360 (63 hand + 99 pose + 198 face padding)

# Metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

# Class encoding
LabelEncoder: 22 Nepali gesture classes (0-21)
```

#### MediaPipe Specification
```python
# Hand Landmarker
- 21 keypoints per hand
- 3 coordinates per keypoint (x, y, z)
- Total: 63 features per frame
- Task file: hand_landmarker.task

# Pose Landmarker
- 33 keypoints (full body skeleton)
- 3 coordinates per keypoint
- Total: 99 features per frame
- Task file: pose_landmarker.task

# Face Detector
- 468 keypoints per face (currently disabled)
- Replaced with 198 zero-padding for consistency
- Can be re-enabled for enhanced detection
```

---

## 4.10.4 GPU and CPU Utilization

### GPU Status and Limitations
- **TensorFlow GPU Support**: NOT AVAILABLE
  - Reason: Native Windows TensorFlow >= 2.11 has no GPU support
  - Message: "TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11"
  
### CPU Utilization Strategy
- **CPU Cores**: 8 cores available
- **Training Parallelization**: Limited by TensorFlow threading
- **Inference Parallelization**: Single-threaded or multi-threaded via NumPy/TensorFlow defaults
- **Batch Processing**: CPU-optimized batches (batch_size configurable, typically 32)

### Alternative GPU Solutions (Not Implemented)
1. **WSL2 (Windows Subsystem for Linux 2)**
   - Full CUDA/cuDNN support for GPU acceleration
   - Recommended for production training

2. **TensorFlow-DirectML Plugin**
   - Alternative: `tensorflow-directml` package
   - Enables DirectML backend on Windows
   - Implementation: Install separately, automatic fallback

3. **Cloud GPU Services**
   - Azure ML, Google Colab, AWS SageMaker
   - Suitable for large-scale training beyond local capabilities

### Performance Characteristics (CPU-based)
- **Training Time**: ~5-10 minutes per model (CPU-only)
- **Inference Time**: 
  - Static: ~50-100ms per frame
  - Dynamic (temporal): ~200-300ms per sequence (40 frames)
- **Real-time Performance**: ~10-12 FPS achievable with temporal stride=2
- **Memory Bottleneck**: RAM (11.85 GB available) is limiting factor
  - Dataset: ~30 MB
  - Models: ~13 MB
  - Working memory during training: ~500 MB - 1 GB per batch

### Optimization Techniques
```python
# 1. Inference Stride (Section 4.11.4)
inference_stride = 2  # Process every 2nd frame
result: ~2x speedup in frame processing

# 2. Batch Size Optimization
batch_size = 32  # Default for CPU
# Larger batches = fewer iterations, less overhead
# Smaller batches = lower memory usage

# 3. Early Stopping (Section 4.8)
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
# Result: Prevents overfitting, reduces training time by ~30-40%

# 4. Model Checkpointing
tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_loss',
    save_best_only=True
)
# Result: Captures best model during training, automatic recovery
```

### Memory Management
```python
# CPU Memory Allocation
# TensorFlow uses CPU memory as needed (no pre-allocation)
# Memory usage scales with:
# - Batch size (larger = more memory)
# - Sequence length (40 frames × 360 features = ~115 KB per sequence)
# - Model size (GRU: ~230 KB, BiGRU_Attention: ~250 KB)

# Data augmentation creates 5× expanded training set in-memory:
# 176 original × 5 = 880 sequences
# 880 × 115 KB ≈ 100 MB (during augmentation phase)
```

---

## 4.10.5 Model Serialization and Storage

### Model Format and Architecture

#### Primary Models (HDF5 Format - .h5)
```
models/final/
├── best_model.h5               (Selected best model after final training)
├── GRU_best.h5                 (2-layer GRU, 64→32 units)
├── BiGRU_best.h5               (Bidirectional GRU)
├── BiGRU_Attention_best.h5     (BiGRU + Self-Attention)
├── meta.json                   (Training configuration metadata)
└── label_encoder.pkl           (Gesture class encoding: 22 classes)

models/dynamic/
├── nsl_gru_dynamic_model.h5    (Production GRU model for inference)
├── meta.json                   (Model metadata and training config)
└── label_encoder.pkl           (Label encoding for 22 gestures)
```

#### Model Storage Details

| Model | Size | Architecture | Purpose |
|-------|------|--------------|---------|
| **best_model.h5** | ~2.1 MB | GRU (2-layer: 64, 32 units) | Primary production model |
| **GRU_best.h5** | ~2.1 MB | Same as best_model | Baseline GRU architecture |
| **BiGRU_best.h5** | ~2.3 MB | BiGRU (64, 32 units) | Bidirectional comparison |
| **BiGRU_Attention_best.h5** | ~2.4 MB | BiGRU + AttentionLayer | Advanced attention-based |
| **nsl_gru_dynamic_model.h5** | ~2.1 MB | GRU variant | Runtime inference model |

### Serialization Mechanism

#### HDF5 Format (.h5)
```python
# Saving Models
model.save('best_model.h5')

# Loading Models
from tensorflow.keras.utils import custom_object_scope
from improved_model import AttentionLayer, LabelSmoothingLoss, TemporalConsistencyLoss

custom_objects = {
    'AttentionLayer': AttentionLayer,
    'LabelSmoothingLoss': LabelSmoothingLoss,
    'TemporalConsistencyLoss': TemporalConsistencyLoss
}

with custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('best_model.h5', compile=False)

# Benefits:
# - Single-file format
# - Preserves model architecture, weights, training state
# - Compatible across platforms
# - Recommended for TensorFlow/Keras models
```

#### Pickle Format (.pkl)
```python
# Label Encoder Serialization
import joblib

# Save
joblib.dump(label_encoder, 'label_encoder.pkl')

# Load
label_encoder = joblib.load('label_encoder.pkl')

# Static Model Serialization (scikit-learn)
joblib.dump(static_classifier, 'static_classifier.pkl')
static_classifier = joblib.load('static_classifier.pkl')

# Benefits:
# - Preserves class labels and mapping
# - Efficient binary format
# - Fast I/O operations
# - Standard scikit-learn serialization method
```

### Metadata Storage (JSON Format)

```json
{
  "timestamp": "2026-05-11T10:30:00",
  "model_name": "best_model",
  "architecture": "GRU",
  "input_shape": [40, 360],
  "output_classes": 22,
  "class_names": ["aapatkal", "adhikari", ..., "umer"],
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "Adam",
    "initial_learning_rate": 0.001,
    "loss_function": "TemporalConsistencyLoss",
    "label_smoothing": 0.1,
    "temporal_weight": 0.05
  },
  "augmentation_config": {
    "window_size": 40,
    "augmentation_factor": 5,
    "augmentation_types": [
      "gaussian_noise (σ=0.01)",
      "temporal_shift (1-5 frames)",
      "crop_and_pad (80%)",
      "feature_dropout (10%)",
      "mixup (α=0.2)"
    ]
  },
  "performance_metrics": {
    "test_accuracy": 0.9107,
    "macro_auc": 0.9908,
    "robustness_degradation_50_occlusion": -0.1176
  },
  "dataset_info": {
    "total_sequences": 276,
    "gesture_classes": 22,
    "train_samples": 176,
    "validation_samples": 44,
    "test_samples": 56,
    "sequence_length": 40,
    "features_per_frame": 360
  }
}
```

### Output Storage (Training and Evaluation Results)

#### Training Outputs
```
outputs/final/
├── best_model_test_report.txt        (Test set classification report)
├── best_model_test_cm.png            (Test confusion matrix)
├── GRU_best_report.txt               (Validation metrics)
├── GRU_best_cm.png                   (Validation confusion matrix)
├── GRU_best_test_report.txt          (Test metrics)
├── GRU_best_test_cm.png              (Test confusion matrix)
├── BiGRU_best_report.txt
├── BiGRU_best_cm.png
├── BiGRU_Attention_best_report.txt
└── BiGRU_Attention_best_cm.png
```

#### Evaluation Outputs (Section 4.8)
```
outputs/advanced_eval/best_model/
├── framework_metadata.json           (4.8.1: Setup info)
├── classification_report.txt         (4.8.2: Metrics)
├── basic_metrics.png                 (4.8.2: Visualization)
├── basic_metrics.json
├── confusion_matrix_raw.png          (4.8.3: Raw CM)
├── confusion_matrix_normalized.png   (4.8.3: Normalized CM)
├── confusion_matrix.csv              (4.8.3: CSV export)
├── confusion_matrix_stats.json
├── roc_curves.png                    (4.8.4: ROC curves)
├── auc_scores.json                   (4.8.4: AUC metrics)
├── class_performance.png             (4.8.5: Per-class metrics)
├── class_metrics.csv                 (4.8.5: Detailed metrics)
├── top_misclassifications.csv        (4.8.6: Error analysis)
├── top_misclassifications.png        (4.8.6: Visualization)
├── occlusion_robustness.png          (4.8.7: Robustness test)
├── occlusion_results.json            (4.8.7: Degradation metrics)
├── stability_analysis.png            (4.8.8: Stability metrics)
└── stability_metrics.json            (4.8.8: Latency, consistency)
```

### Data Format Specifications

#### NumPy Data Format (.npy) - Input
```python
# Dataset sequence format:
# Shape: (40, 360)
# 40 frames × 360 features per frame

# Features per frame:
# [0:63]     = Hand landmarks (21 keypoints × 3 coords)
# [63:162]   = Pose landmarks (33 keypoints × 3 coords)
# [162:360]  = Face padding (198 zeros, optional face features)

# Storage location:
dataset_word/landmarks/final/{gesture_class}/
├── sequence_0.npy  # One .npy file per recorded sequence
├── sequence_1.npy
├── sequence_2.npy
└── ...
```

#### Training State Checkpointing
```python
# EarlyStopping Configuration
monitor='val_loss'
patience=15          # Stop after 15 epochs with no improvement
restore_best_weights=True  # Automatically restore best weights

# Model Checkpoint
save_best_only=True  # Only save if improvement detected
monitor='val_loss'   # Save when validation loss improves
mode='min'           # Lower is better
```

### Serialization Best Practices

#### 1. Custom Layer Serialization
```python
# AttentionLayer requires custom_objects registration
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_config(self):
        return super().get_config()  # Required for serialization
```

#### 2. Custom Loss Function Serialization
```python
class TemporalConsistencyLoss(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, temporal_weight=0.05, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.temporal_weight = temporal_weight
    
    def get_config(self):
        # Required for proper serialization and deserialization
        return {
            'label_smoothing': self.label_smoothing,
            'temporal_weight': self.temporal_weight,
            **super().get_config()
        }
```

#### 3. Loading with Custom Objects
```python
# Always use custom_object_scope for safe loading
custom_objects = {
    'AttentionLayer': AttentionLayer,
    'LabelSmoothingLoss': LabelSmoothingLoss,
    'TemporalConsistencyLoss': TemporalConsistencyLoss
}

with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_path, compile=False)
```

### Storage Efficiency

| Component | Size | Efficiency Notes |
|-----------|------|------------------|
| **Model Weights** | ~2.1 MB | HDF5 compression enabled by default |
| **Training Data** | ~30 MB | 276 sequences × ~115 KB each |
| **Augmented Data** | ~150 MB | 5× expansion created dynamically during training |
| **Evaluation Outputs** | ~0.69 MB | Metrics and visualizations (PNG + JSON + CSV) |
| **Total Archive Size** | ~50 MB | Suitable for cloud storage, version control |

### Reproducibility and Version Control

#### Git Integration
```
.gitignore includes:
- *.h5 (model files - too large)
- *.npy (dataset files - binary)
- outputs/ (generated outputs)
- venv/ (virtual environment)
- __pycache__/ (Python cache)

Tracked files:
- scripts/ (Python code)
- notebooks/ (Training procedures)
- improved_model.py (Architecture definitions)
- meta.json (Configuration metadata)
```

#### Versioning Strategy
```python
# File naming convention for versioning
nsl_gru_dynamic_model_{timestamp}_v1.0.h5
# Format: {model_name}_{timestamp}_v{version}.h5

# Metadata includes:
- Training date/time
- TensorFlow version
- Dataset version
- Hyperparameter configuration
- Performance metrics at save time
```

---

## Summary Table

| Aspect | Specification |
|--------|---------------|
| **CPU** | 8-core Intel, 11.85 GB RAM |
| **GPU** | None (Windows TensorFlow limitation) |
| **OS** | Windows 10 (10.0.26200) |
| **Python** | 3.11.8 |
| **TensorFlow** | 2.21.0 |
| **Main Framework** | Keras (sequential/functional API) |
| **Model Format** | HDF5 (.h5) with custom objects |
| **Storage Format** | HDF5 (models), Pickle (encoders), JSON (metadata) |
| **Training Time** | ~5-10 min/model (CPU-based) |
| **Inference Speed** | 50-100ms (static), 200-300ms (dynamic) |
| **Real-time FPS** | ~10-12 FPS with stride=2 |

