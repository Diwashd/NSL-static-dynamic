import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
from collections import deque
import json
import tensorflow as tf
from tensorflow.keras import layers
from statistics import mode, StatisticsError

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure Streamlit
st.set_page_config(page_title="NSL Hand Gesture Detector", page_icon="🤚", layout="wide")
st.title("🤚 NSL Hand Gesture Detector - Real-time Verification")

# Import refactored helpers from nsl_app package
from nsl_app import REPO_ROOT
from nsl_app.models import (
    AttentionLayer,
    label_smoothing_loss,
)
from nsl_app.models import (
    get_available_static_models as _models_get_available_static_models,
    get_available_dynamic_models as _models_get_available_dynamic_models,
    load_selected_static_model as _models_load_selected_static_model,
    load_selected_dynamic_model as _models_load_selected_dynamic_model,
)
from nsl_app.mediapipe_utils import load_detectors
from nsl_app.visuals import draw_landmarks, draw_pose_skeleton
from nsl_app.utils import (
    get_nepali_character,
    get_gru_class_name,
    extract_normalized_landmarks,
)

# ============ MODEL SELECTOR ============
st.sidebar.title("⚙️ Model Configuration")
model_type = st.sidebar.radio(
    "Select Model Type",
    ["Static (Single Frame)", "Dynamic (Sequence)"],
    help="Static: Single-frame landmarks. Dynamic: Hand movement sequences."
)

@st.cache_resource
def get_available_static_models():
    return _models_get_available_static_models(REPO_ROOT)

@st.cache_resource
def get_available_dynamic_models():
    return _models_get_available_dynamic_models(REPO_ROOT)

# Custom layer and loss moved to nsl_app.models

# Path resolution moved to nsl_app.config (REPO_ROOT)

# ============ MODEL LOADING ============
@st.cache_resource
def load_model_and_detector():
    # Load sklearn static model (if present) and mediapipe detectors
    model_path = os.path.join(REPO_ROOT, "notebooks", "nsl_model_v2.pkl")
    try:
        model = joblib.load(model_path)
    except Exception:
        st.warning(f"Could not load sklearn model from {model_path}")
        model = None

    hand_detector_static, pose_detector, face_detector = load_detectors(REPO_ROOT, num_hands=1)
    hand_detector_dynamic, _, _ = load_detectors(REPO_ROOT, num_hands=2)
    return model, hand_detector_static, hand_detector_dynamic, pose_detector, face_detector

@st.cache_resource
def load_selected_static_model(model_name):
    model, scaler = _models_load_selected_static_model(REPO_ROOT, model_name)
    if model is not None:
        st.sidebar.success(f"✅ Loaded: {model_name}")
    else:
        st.sidebar.error(f"Failed to load {model_name}")
    return model, scaler

@st.cache_resource
def load_selected_dynamic_model(model_name):
    dynamic_model, config = _models_load_selected_dynamic_model(REPO_ROOT, model_name)
    if dynamic_model is not None:
        st.sidebar.success(f"✅ Loaded: {model_name}")
        return dynamic_model, config
    else:
        st.sidebar.error(f"Failed to load {model_name}")
        return None, None

model, hand_detector_static, hand_detector_dynamic, pose_detector, face_detector = load_model_and_detector()

# Load models based on selection
if model_type == "Static (Single Frame)":
    static_models = get_available_static_models()
    if static_models:
        model_list = list(static_models.keys())
        # Default to nsl_model_v2 from train.ipynb if available
        default_idx = 0
        if "nsl_model_v2" in model_list:
            default_idx = model_list.index("nsl_model_v2")
        
        selected_static = st.sidebar.selectbox(
            "Select Static Model",
            model_list,
            index=default_idx,
            help="Train.ipynb: Single-frame hand landmark classification"
        )
        static_model, static_scaler = load_selected_static_model(selected_static)
        gru_model, gru_config = None, None
    else:
        st.sidebar.warning("❌ No static models found. Run: notebooks/train.ipynb")
        static_model, static_scaler = None, None
        gru_model, gru_config = None, None
else:
    dynamic_models = get_available_dynamic_models()
    if dynamic_models:
        model_list = list(dynamic_models.keys())
        # Default selection: prefer GRU_best, GRU, then best_model (Default)
        default_idx = 0
        preferred = ["GRU_best", "GRU", "GRU.BEST", "best_model (Default)", "best_model"]
        for name in preferred:
            if name in model_list:
                default_idx = model_list.index(name)
                break

        selected_dynamic = st.sidebar.selectbox(
            "Select Dynamic Model",
            model_list,
            index=default_idx,
            help="Dynamic_training.ipynb: Sequence-based hand gesture recognition"
        )
        gru_model, gru_config = load_selected_dynamic_model(selected_dynamic)
        static_model, static_scaler = None, None
    else:
        st.sidebar.warning("❌ No dynamic models found. Run: scripts/dynamic_training.ipynb")
        gru_model, gru_config = None, None
        static_model, static_scaler = None, None

# Mapping, index lists and helpers moved to nsl_app.utils
# Drawing utilities moved to nsl_app.visuals

# Landmark extraction helpers moved to nsl_app.utils

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()
    frame_info = st.empty()

with col2:
    st.subheader("Prediction & Data")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    latency_placeholder = st.empty()
    data_placeholder = st.empty()

# Detection Mode Selection
st.markdown("---")
st.subheader("🎯 Select Detection Mode")

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    static_btn = st.button("📷 Detect Static Sign Language", key="static_btn", use_container_width=True)

with mode_col2:
    # Require both a loaded GRU model and a hand detector Task to enable dynamic detection
    if gru_model is not None and hand_detector_dynamic is not None and pose_detector is not None:
        dynamic_btn = st.button("🎬 Start Detecting Dynamic Language", key="dynamic_btn", use_container_width=True)
    else:
        if gru_model is None:
            st.warning("⚠️ GRU model not available for dynamic detection")
        if hand_detector_dynamic is None:
            st.warning("⚠️ Hand landmarker not loaded. Place `hand_landmarker.task` in repo root or use scripts/download_hand_task.py to fetch it.")
        if pose_detector is None:
            st.warning("⚠️ Pose landmarker not loaded. Place `pose_landmarker.task` in repo root.")
        dynamic_btn = False

# STATIC DETECTION MODE
if static_btn:
    st.info("📷 **Static Sign Language Detection** - Single frame classification")
    confidence_threshold = 0.50  # Only accept predictions with >50% confidence
    st.write(f"Confidence Threshold: {confidence_threshold*100:.0f}%")
    static_predictor = static_model if static_model is not None else model
    if hand_detector_static is None:
        st.error("Hand landmarker not loaded. Place `hand_landmarker.task` in repo root.")
    elif static_predictor is None:
        st.error("Static model not available. Train or select a static model.")
    else:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Unable to access webcam!")
        else:
            st.info("Press 'Q' in the video window to stop. Make sure your hand is visible and well-lit.")
            debug_info = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = time.time()

                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect hand landmarks
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = hand_detector_static.detect(mp_image)

                prediction = None
                confidence = None
                landmarks_data = None
                latency_ms = None
                hand_detected = False

                # Debug info
                debug_text = f"Hand landmarks detected: {len(results.hand_landmarks) if results.hand_landmarks else 0}"

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand_detected = True
                    hand = results.hand_landmarks[0]

                    # Draw landmarks on frame
                    frame = draw_landmarks(frame, hand)

                    # Extract normalized landmarks
                    landmarks, hand_size = extract_normalized_landmarks(hand)

                    debug_text += f" | Hand size: {hand_size:.3f} | Features: {len(landmarks)}"

                    if landmarks and hand_size > 0.01:
                        # Make prediction using static model
                        prediction = static_predictor.predict([landmarks])[0]

                        # Get prediction probabilities
                        prediction_proba = static_predictor.predict_proba([landmarks])[0]
                        confidence = np.max(prediction_proba)

                        # Only keep prediction if confidence meets threshold
                        if confidence < confidence_threshold:
                            prediction = None
                            confidence = None
                        else:
                            confidence = confidence * 100  # Convert to percentage
                            landmarks_data = landmarks

                # Calculate frame processing latency
                latency_ms = (time.time() - frame_start_time) * 1000

                if prediction:
                    # Draw prediction on frame
                    nepali_char = get_nepali_character(prediction)
                    text = f"Sign: {nepali_char} ({confidence:.1f}%)"
                    cv2.putText(frame, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Hand Size: {hand_size:.3f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Draw latency on frame
                    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Display video frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                debug_info.caption(f"🔍 Debug: {debug_text}")

                # Update prediction info
                if prediction:
                    nepali_char = get_nepali_character(prediction)
                    prediction_placeholder.success(f"**Prediction:** {nepali_char}")
                    confidence_placeholder.info(f"**Confidence:** {confidence:.2f}%")
                    latency_placeholder.metric("Latency", f"{latency_ms:.1f}ms")

                    # Show normalized landmarks
                    with data_placeholder.expander("📊 Normalized Landmark Data"):
                        st.write(f"**Total Features:** {len(landmarks_data)}")
                        st.write(f"**Hand Size Factor:** {hand_size:.4f}")

                        # Display as formatted columns
                        landmark_cols = st.columns(3)
                        for i in range(0, len(landmarks_data), 3):
                            with landmark_cols[0 if i % 9 == 0 else 1 if i % 9 == 3 else 2]:
                                st.write(f"**LM {i//3}:** X={landmarks_data[i]:.4f}")
                                st.write(f"Y={landmarks_data[i+1]:.4f}")
                                st.write(f"Z={landmarks_data[i+2]:.4f}")
                elif hand_detected:
                    prediction_placeholder.warning("❌ **No sign detected** - Confidence below threshold or invalid gesture")
                else:
                    prediction_placeholder.warning("❌ **No hand detected** - Position your hand clearly in the camera")

                # Check for stop condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            st.success("✅ Static detection stopped!")

# DYNAMIC DETECTION MODE (GRU-based)
elif dynamic_btn and gru_model is not None:
    st.info("🎬 **Dynamic Sign Language Detection** - Temporal sequence analysis")
    expected_feature_len = 360
    
    # Use the sequence length from model's input shape
    sequence_length = gru_config.get("config", {}).get("augmentation_config", {}).get("window_size", 40) if gru_config else 40
    min_sequence_frames = max(12, int(sequence_length * 0.4))
    inference_stride = 2
    confidence_threshold = 0.60  # Only accept predictions with >60% confidence
    
    # ====== 4.11.3 TEMPORAL PREDICTION SMOOTHING CONFIG ======
    prediction_history_size = 3  # Majority voting window
    ema_alpha = 0.3  # Exponential Moving Average factor (0.0-1.0)
    temporal_consistency_threshold = 0.5  # Minimum agreement for consistency
    
    st.write(
        f"Model Window: {sequence_length} frames | "
        f"Fast Trigger: {min_sequence_frames}+ frames | "
        f"Confidence Threshold: {confidence_threshold*100:.0f}% | "
        f"Smoothing: Voting={prediction_history_size} EMA={ema_alpha}"
    )
    st.info("Prediction starts early with temporal smoothing for stable, jitter-free detection")

    def resample_sequence_for_model(sequence, target_length):
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got shape {arr.shape}")

        n_frames, n_features = arr.shape
        if n_frames == target_length:
            return arr
        if n_frames <= 1:
            return np.repeat(arr, target_length, axis=0)

        src_idx = np.arange(n_frames, dtype=np.float32)
        dst_idx = np.linspace(0, n_frames - 1, target_length, dtype=np.float32)
        resampled = np.empty((target_length, n_features), dtype=np.float32)
        for feat in range(n_features):
            resampled[:, feat] = np.interp(dst_idx, src_idx, arr[:, feat])
        return resampled
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Unable to access webcam!")
    else:
        st.info(f"Press 'Q' to stop. Make sure your hand is visible and well-lit.")
        debug_info = st.empty()
        
        landmark_sequence = deque(maxlen=sequence_length)
        frame_count = 0
        last_prediction = None
        last_confidence = 0
        prediction_reset_counter = 0
        
        # ====== 4.11.3 TEMPORAL SMOOTHING BUFFERS ======
        prediction_history = deque(maxlen=prediction_history_size)  # Last N predictions
        confidence_history = deque(maxlen=prediction_history_size)  # Last N confidence scores
        smoothed_confidence = 0.0  # EMA-smoothed confidence
        frame_times = deque(maxlen=30)  # For FPS calculation
        frame_start_total = time.time()
        
        # ====== 4.11.4 LATENCY TRACKING ======
        prediction_latencies = deque(maxlen=30)
        landmark_extraction_latencies = deque(maxlen=30)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            frame_count += 1
            frame_times.append(frame_start_time)
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect multi-modal landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_results = hand_detector_dynamic.detect(mp_image)
            pose_result = pose_detector.detect(mp_image) if pose_detector else None

            prediction = None
            confidence = None
            hand_detected = False

            hands = hand_results.hand_landmarks if hand_results and hand_results.hand_landmarks else []
            hands_detected = len(hands)
            pose_landmarks = pose_result.pose_landmarks[0] if pose_result and pose_result.pose_landmarks else None
            pose_detected = pose_landmarks is not None

            debug_text = f"Hands detected: {hands_detected} | Pose detected: {'Yes' if pose_detected else 'No'}"

            if hands_detected >= 2 and pose_detected:
                hand_detected = True

                for hand in hands[:2]:
                    frame = draw_landmarks(frame, hand)
                frame = draw_pose_skeleton(frame, pose_landmarks)

                hand_size = math.sqrt(
                    (hands[0][17].x - hands[0][5].x) ** 2 +
                    (hands[0][17].y - hands[0][5].y) ** 2
                )

                # Extract full landmarks (360 features): two hands + pose + padding
                landmark_extract_start = time.time()
                try:
                    full_landmarks = []

                    for idx in range(2):
                        if idx < hands_detected:
                            for lm in hands[idx]:
                                full_landmarks.extend([lm.x, lm.y, lm.z])
                        else:
                            full_landmarks.extend([0] * 63)

                    for lm in pose_landmarks:
                        full_landmarks.extend([lm.x, lm.y, lm.z])

                    full_landmarks = full_landmarks[:expected_feature_len]
                    full_landmarks.extend([0] * (expected_feature_len - len(full_landmarks)))

                    landmark_sequence.append(np.asarray(full_landmarks, dtype=np.float32).tolist())
                    landmark_extraction_latencies.append((time.time() - landmark_extract_start) * 1000)
                    debug_text += f" | Hand size: {hand_size:.3f} | Features: {len(full_landmarks)}"
                except Exception as e:
                    debug_text += f" | Landmark extraction failed: {str(e)}"
                    st.warning(f"Error extracting landmarks: {str(e)}")

                # Start predicting early; resample buffered frames to the model window size
                if len(landmark_sequence) >= min_sequence_frames and (frame_count % inference_stride == 0):
                    model_sequence = resample_sequence_for_model(list(landmark_sequence), sequence_length)
                    sequence_array = np.expand_dims(model_sequence, axis=0)
                    try:
                        prediction_inference_start = time.time()
                        prediction = gru_model.predict(sequence_array, verbose=0)
                        prediction_latencies.append((time.time() - prediction_inference_start) * 1000)

                        prediction_class = np.argmax(prediction[0])
                        confidence = np.max(prediction[0])

                        # ====== 4.11.3 TEMPORAL SMOOTHING: COLLECT PREDICTION HISTORY ======
                        prediction_history.append(prediction_class)
                        confidence_history.append(confidence)

                        # Exponential Moving Average (EMA) smoothing for confidence
                        smoothed_confidence_var = ema_alpha * confidence + (1 - ema_alpha) * (smoothed_confidence if smoothed_confidence > 0 else confidence)
                        smoothed_confidence = smoothed_confidence_var

                        # Majority voting across prediction history
                        if len(prediction_history) >= 2:
                            try:
                                voted_prediction = mode(prediction_history)
                                # Calculate consistency score: fraction of votes for winner
                                vote_counts = {p: list(prediction_history).count(p) for p in set(prediction_history)}
                                max_votes = max(vote_counts.values())
                                consistency = max_votes / len(prediction_history)
                            except StatisticsError:
                                voted_prediction = prediction_class
                                consistency = 1.0
                        else:
                            voted_prediction = prediction_class
                            consistency = 1.0

                        # Accept prediction if smoothed confidence meets threshold AND temporal consistency is high
                        if smoothed_confidence >= confidence_threshold and consistency >= temporal_consistency_threshold:
                            last_prediction = voted_prediction
                            last_confidence = smoothed_confidence * 100
                            prediction_reset_counter = 0
                        else:
                            last_prediction = None
                            last_confidence = 0

                    except Exception as e:
                        st.warning(f"Prediction error: {e}")
            else:
                last_prediction = None
                last_confidence = 0
                debug_text += " | Waiting for two hands + pose"
            
            # Calculate frame processing latency
            latency_ms = (time.time() - frame_start_time) * 1000
            
            # ====== 4.11.4 FPS CALCULATION ======
            if len(frame_times) >= 2:
                time_diff = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / time_diff if time_diff > 0 else 0
            else:
                fps = 0
            
            # Calculate average latencies
            avg_prediction_latency = np.mean(prediction_latencies) if prediction_latencies else 0
            avg_landmark_latency = np.mean(landmark_extraction_latencies) if landmark_extraction_latencies else 0
            
            # Draw information on frame
            cv2.putText(frame, f"Frames: {len(landmark_sequence)} | Ready: {min_sequence_frames}+ | FPS: {fps:.1f}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if last_prediction is not None and last_confidence >= confidence_threshold * 100:
                class_name = get_gru_class_name(last_prediction, gru_config, REPO_ROOT)
                text = f"Sign: {class_name} ({last_confidence:.1f}%)"
                cv2.putText(frame, text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                prediction_reset_counter += 1
            
            cv2.putText(frame, f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display video frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)
            debug_info.caption(f"🔍 Debug: {debug_text}")
            
            if last_prediction is not None and last_confidence >= confidence_threshold * 100:
                class_name = get_gru_class_name(last_prediction, gru_config, REPO_ROOT)
                prediction_placeholder.success(f"**🎯 Detected Sign:** {class_name.upper()}")
                confidence_placeholder.info(f"**Confidence:** {last_confidence:.2f}%")
            else:
                if len(landmark_sequence) > 0:
                    prediction_placeholder.info(
                        f"**⏳ Collecting frames:** {len(landmark_sequence)}/{min_sequence_frames} (fast trigger)"
                    )
                else:
                    prediction_placeholder.warning("❌ **No sign detected** - Confidence too low or missing two hands + pose")
            
            if hand_detected:
                frame_info.info(
                    f"✅ Two hands + pose detected | "
                    f"Sequence: {len(landmark_sequence)} | Ready at {min_sequence_frames}+"
                )
            else:
                frame_info.warning(
                    f"❌ Need two hands + pose | Hands: {hands_detected} | Pose: {'Yes' if pose_detected else 'No'} | "
                    f"Sequence: {len(landmark_sequence)} | Ready at {min_sequence_frames}+"
                )
            
            # ====== 4.11.3-4.11.4 DISPLAY COMPREHENSIVE METRICS ======
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                latency_placeholder.metric("Latency", f"{latency_ms:.1f}ms")
                st.metric("Prediction Latency", f"{avg_prediction_latency:.1f}ms")
            
            with metrics_col2:
                st.metric("FPS", f"{fps:.1f}")
                st.metric("Landmark Latency", f"{avg_landmark_latency:.1f}ms")
            
            with metrics_col3:
                if len(confidence_history) > 0:
                    st.metric("EMA Confidence", f"{smoothed_confidence*100:.1f}%")
                    consistency_pct = (max(list(prediction_history).count(p) for p in set(prediction_history)) / len(prediction_history) * 100 if len(prediction_history) > 0 else 0)
                    st.metric("Consistency", f"{consistency_pct:.0f}%")
                else:
                    st.metric("EMA Confidence", "N/A")
                    st.metric("Consistency", "N/A")
            
            # Check for stop condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        st.success("✅ Dynamic detection stopped!")

st.markdown("---")
st.markdown("""
### 📖 How to Use:

#### **📷 Static Sign Language Detection:**
1. Click **Detect Static Sign Language** button
2. Allow webcam access when prompted
3. Position your hand in the camera view
4. The app will display:
   - **Live video** with hand landmarks overlay
   - **Single-frame prediction** of the sign gesture
   - **Confidence** score based on frame analysis
   - **Normalized landmark data** for verification
5. Press 'Q' to stop

#### **🎬 Dynamic Sign Language Detection (with Temporal Smoothing):**
1. Click **Start Detecting Dynamic Language** button
2. Allow webcam access when prompted
3. Use **both hands** and keep your upper body visible for pose tracking
4. Perform the sign with both hands (takes ~1 second)
5. The app collects multiple frames and analyzes temporal patterns using:
   - **🎯 Majority Voting** (4.11.3): Consensus across last 3 predictions
   - **📊 EMA Smoothing**: Exponential Moving Average confidence filtering
   - **⏱️ Temporal Consistency**: Validates prediction agreement across frames
6. The display shows:
   - **Live video** with landmarks and FPS counter
   - **Frame sequence progress** (e.g., 15/30 frames)
   - **Gesture prediction** with temporal analysis
   - **Confidence** score (EMA-smoothed)
   - **Consistency** score (prediction agreement %)
   - **FPS** (frames per second)
   - **Latency breakdown** (prediction + landmark extraction)
7. Press 'Q' to stop

### ⚙️ Model Information:
- **Static Model:** Single-frame classifier using scikit-learn
- **Dynamic Model:** GRU (Gated Recurrent Unit) for temporal sequences
- **4.11.3 Temporal Smoothing:** EMA + Majority Voting + Consistency Checking
- **4.11.4 Latency Optimization:** Per-component timing with stride-based inference

### 📈 Smoothing Parameters (Configurable):
- **Prediction History Size:** 3 frames (majority voting window)
- **EMA Alpha:** 0.3 (confidence smoothing factor)
- **Temporal Consistency Threshold:** 0.5 (50% minimum agreement)
- **Inference Stride:** 2 (skip frames for performance)
- **Min Sequence Frames:** 12+ (early trigger threshold)
""")
