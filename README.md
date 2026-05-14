# NSL Static + Dynamic Landmark Thesis

Nepali Sign Language (NSL) recognition using MediaPipe landmarks with two pipelines:
- Static (single-frame) classification with a scikit-learn MLP.
- Dynamic (sequence) classification with GRU-based models.

The repo includes a Streamlit demo for real-time verification, training notebooks, and comprehensive evaluation scripts (accuracy, confusion matrices, ROC/AUC, occlusion tests, and stability analysis).

## Highlights
- Hand, pose, and face-aware landmark extraction (face features optionally padded).
- Side-by-side static vs dynamic modeling and evaluation.
- Advanced evaluation suite aligned with thesis sections.
- Streamlit UI for live webcam inference with selectable models.

## Project Structure
- app.py: Streamlit real-time demo.
- nsl_app/: Shared utilities, model loaders, and MediaPipe helpers.
- scripts/: Training/evaluation scripts (dynamic training, advanced evaluation, augmentation, etc.).
- notebooks/: Training notebooks (static and dynamic).
- dataset_word/: Dynamic sequence dataset (NumPy sequences).
- nsl_landmarks.csv, nsl_landmarks_v2.csv: Static landmark CSVs.
- models/: Saved models (static .pkl, dynamic .h5).
- outputs/: Reports, figures, and evaluation artifacts.
- training environment.md: Full training environment and system config.

## Quick Start (Windows)
1) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

2) Install dependencies
```bash
pip install tensorflow==2.21.0 mediapipe==0.10.33 opencv-python==4.13.0 \
    scikit-learn==1.8.0 numpy==2.4.3 pandas==2.3.3 joblib \
    streamlit matplotlib seaborn
```

3) Ensure MediaPipe task files exist in the repo root
- hand_landmarker.task
- pose_landmarker.task
- face_landmarker.task (optional)

If you only need the hand task, run:
```bash
python scripts/download_hand_task.py
```

## Run the Streamlit Demo
```bash
streamlit run app.py
```
- Choose Static or Dynamic in the sidebar.
- Select a model if multiple are available.
- Use a webcam and ensure good lighting.

## Training and Evaluation
Static (single-frame) training:
```bash
jupyter notebook notebooks/static_train.ipynb
```

Dynamic (sequence) training:
```bash
jupyter notebook scripts/dynamic_training.ipynb
```

Advanced evaluation (metrics, ROC/AUC, occlusion, stability):
```bash
python scripts/advanced_evaluation.py
```

Evaluate all saved models (static + dynamic):
```bash
python scripts/evaluate_all_models.py
```

## Data Notes
- Dynamic sequences: dataset_word/landmarks/final/ (22 classes, .npy files).
- Static CSVs: nsl_landmarks.csv and nsl_landmarks_v2.csv (36 classes).

## Outputs
Evaluation artifacts are written under outputs/ (reports, confusion matrices, ROC curves, robustness tests, and stability metrics).

## Large Files and Git
This project includes large datasets and model files. GitHub rejects files over 100 MB and warns for files over 50 MB. Use Git LFS or exclude these files via .gitignore if needed.

## License
- Dataset license: dataset/LICENSE
- Project code: see repository policy (add a license file if required).
