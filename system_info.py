#!/usr/bin/env python3
"""System and Environment Information for NSL Gesture Recognition Thesis"""

import sys
import os
import platform
import psutil
import tensorflow as tf
import sklearn
import mediapipe
import numpy as np
import pandas as pd
import joblib
import cv2

print("=" * 80)
print("4.10 TRAINING ENVIRONMENT AND SYSTEM CONFIGURATION")
print("=" * 80)

print("\n4.10.1 HARDWARE SPECIFICATIONS")
print("-" * 80)
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print(f"Machine: {platform.machine()}")
print(f"CPU Count: {os.cpu_count()} cores")

# Memory info
vm = psutil.virtual_memory()
print(f"Total RAM: {vm.total / (1024**3):.2f} GB")
print(f"Available RAM: {vm.available / (1024**3):.2f} GB")

print("\n4.10.2 SOFTWARE ENVIRONMENT")
print("-" * 80)
print(f"Operating System: {sys.platform}")
print(f"OS Version: {platform.version()}")
print(f"Python Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")

print("\n4.10.3 PYTHON LIBRARIES AND FRAMEWORKS")
print("-" * 80)
print(f"Python Version: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"MediaPipe: {mediapipe.__version__}")
print(f"OpenCV (cv2): {cv2.__version__}")
print(f"joblib: {joblib.__version__}")

print("\n4.10.4 GPU AND CPU UTILIZATION")
print("-" * 80)
print(f"TensorFlow CUDA Build: {tf.test.is_built_with_cuda()}")

gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU Devices Available: {len(gpu_devices)}")
if gpu_devices:
    for i, gpu in enumerate(gpu_devices):
        print(f"  GPU {i}: {gpu}")
else:
    print("  No GPU devices detected")

cpu_devices = tf.config.list_physical_devices('CPU')
print(f"CPU Devices: {len(cpu_devices)}")

# TensorFlow memory strategy
print(f"\nTensorFlow Memory Growth Config:")
for gpu in gpu_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  Memory growth enabled for GPU")
    except Exception as e:
        print(f"  Memory growth error: {e}")

print("\n4.10.5 MODEL SERIALIZATION AND STORAGE")
print("-" * 80)

# Check model directories
model_dirs = {
    "Final Models": "models/final",
    "Dynamic Models": "models/dynamic",
    "Output Directory": "outputs/final",
    "Evaluation Output": "outputs/advanced_eval",
    "Dataset Location": "dataset_word/landmarks/final"
}

for label, path in model_dirs.items():
    full_path = os.path.join(os.getcwd(), path)
    if os.path.exists(full_path):
        size = sum(f.stat().st_size for f in __import__('pathlib').Path(full_path).rglob('*') if f.is_file())
        print(f"{label}: {full_path}")
        print(f"  Size: {size / (1024**2):.2f} MB")
        
        # List files
        try:
            files = os.listdir(full_path)
            for f in files[:5]:  # Show first 5
                print(f"    - {f}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more files")
        except:
            pass
    else:
        print(f"{label}: {full_path} (not found)")

print("\n" + "=" * 80)
print("End of System Configuration Report")
print("=" * 80)
