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

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSL Hand Gesture Detector",
    page_icon="🤚",
    layout="wide",
)
st.title("🤚 NSL Hand Gesture Detector — Real-time Verification")

# ─────────────────────────────────────────────────────────────
# IMPORTS FROM NSL_APP PACKAGE
# ─────────────────────────────────────────────────────────────
from nsl_app import REPO_ROOT
from nsl_app.models import (
    AttentionLayer,
    label_smoothing_loss,
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

# ─────────────────────────────────────────────────────────────
# SIDEBAR — MODEL SELECTOR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Model Configuration")
model_type = st.sidebar.radio(
    "Select Model Type",
    ["Static (Single Frame)", "Dynamic (Sequence)"],
    help="Static: Single-frame landmarks. Dynamic: Hand movement sequences.",
)


@st.cache_resource
def get_available_static_models():
    return _models_get_available_static_models(REPO_ROOT)


@st.cache_resource
def get_available_dynamic_models():
    return _models_get_available_dynamic_models(REPO_ROOT)


# ─────────────────────────────────────────────────────────────
# DETECTOR + BASE MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_detector():
    model_path = os.path.join(REPO_ROOT, "notebooks", "nsl_model_v2.pkl")
    try:
        model = joblib.load(model_path)
    except Exception:
        st.warning(f"Could not load sklearn model from {model_path}")
        model = None

    hand_detector_static, pose_detector, face_detector = load_detectors(
        REPO_ROOT, num_hands=1
    )
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
    # Show what directory we're scanning so path issues are obvious immediately
    models_dir = os.path.join(REPO_ROOT, "models", "final")
    if not os.path.exists(models_dir):
        st.sidebar.error(
            f"❌ models/final not found.\n"
            f"Expected: {models_dir}\n"
            f"REPO_ROOT = {REPO_ROOT}"
        )
        return None, None

    # List actual files so the user can see what's there if loading fails
    try:
        found = [f for f in os.listdir(models_dir) if f.endswith((".h5", ".keras"))]
    except Exception:
        found = []

    dynamic_model, config = _models_load_selected_dynamic_model(REPO_ROOT, model_name)

    if dynamic_model is not None:
        shape = getattr(dynamic_model, "input_shape", "unknown")
        st.sidebar.success(f"✅ Loaded: {model_name}  |  input shape: {shape}")
        return dynamic_model, config
    else:
        st.sidebar.error(
            f"❌ Failed to load '{model_name}'.\n"
            f"Directory: {models_dir}\n"
            f"Files found: {found if found else 'none'}\n"
            f"Check the terminal / logs for the exact TensorFlow error."
        )
        return None, None


model, hand_detector_static, hand_detector_dynamic, pose_detector, face_detector = (
    load_model_and_detector()
)

# ─────────────────────────────────────────────────────────────
# LOAD SELECTED MODEL
# ─────────────────────────────────────────────────────────────
if model_type == "Static (Single Frame)":
    static_models = get_available_static_models()
    if static_models:
        model_list = list(static_models.keys())
        default_idx = 0
        if "nsl_model_v2" in model_list:
            default_idx = model_list.index("nsl_model_v2")
        selected_static = st.sidebar.selectbox(
            "Select Static Model",
            model_list,
            index=default_idx,
            help="Train.ipynb: Single-frame hand landmark classification",
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
            help="Dynamic_training.ipynb: Sequence-based hand gesture recognition",
        )
        gru_model, gru_config = load_selected_dynamic_model(selected_dynamic)
        static_model, static_scaler = None, None
    else:
        st.sidebar.warning(
            "❌ No dynamic models found. Run: scripts/dynamic_training.ipynb"
        )
        gru_model, gru_config = None, None
        static_model, static_scaler = None, None

# ─────────────────────────────────────────────────────────────
# LAYOUT PLACEHOLDERS
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# DETECTION MODE BUTTONS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Select Detection Mode")

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    static_btn = st.button(
        "📷 Detect Static Sign Language", key="static_btn", use_container_width=True
    )

with mode_col2:
    dynamic_ready = (
        gru_model is not None
        and hand_detector_dynamic is not None
        and pose_detector is not None
    )
    if dynamic_ready:
        dynamic_btn = st.button(
            "🎬 Start Detecting Dynamic Language",
            key="dynamic_btn",
            use_container_width=True,
        )
    else:
        if gru_model is None:
            st.warning("⚠️ GRU model not available for dynamic detection")
        if hand_detector_dynamic is None:
            st.warning(
                "⚠️ Hand landmarker not loaded. Place `hand_landmarker.task` in repo root."
            )
        if pose_detector is None:
            st.warning(
                "⚠️ Pose landmarker not loaded. Place `pose_landmarker.task` in repo root."
            )
        dynamic_btn = False


# ─────────────────────────────────────────────────────────────
# HELPER: sequence resampler
# ─────────────────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════
# STATIC DETECTION MODE
# ═════════════════════════════════════════════════════════════
if static_btn:
    st.info("📷 **Static Sign Language Detection** — Single frame classification")
    confidence_threshold = 0.50
    st.write(f"Confidence Threshold: {confidence_threshold * 100:.0f}%")

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
            st.info("Press 'Q' in the video window to stop.")
            debug_info = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = time.time()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = hand_detector_static.detect(mp_image)

                prediction = None
                confidence = None
                landmarks_data = None
                hand_size = 0.0
                hand_detected = False

                debug_text = (
                    f"Hand landmarks detected: "
                    f"{len(results.hand_landmarks) if results.hand_landmarks else 0}"
                )

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand_detected = True
                    hand = results.hand_landmarks[0]
                    frame = draw_landmarks(frame, hand)

                    landmarks, hand_size = extract_normalized_landmarks(hand)
                    debug_text += f" | Hand size: {hand_size:.3f} | Features: {len(landmarks)}"

                    if landmarks and hand_size > 0.01:
                        prediction = static_predictor.predict([landmarks])[0]
                        prediction_proba = static_predictor.predict_proba([landmarks])[0]
                        confidence = np.max(prediction_proba)

                        if confidence < confidence_threshold:
                            prediction = None
                            confidence = None
                        else:
                            confidence = confidence * 100
                            landmarks_data = landmarks

                latency_ms = (time.time() - frame_start_time) * 1000

                if prediction:
                    nepali_char = get_nepali_character(prediction)
                    text = f"Sign: {nepali_char} ({confidence:.1f}%)"
                    cv2.putText(
                        frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame, f"Hand Size: {hand_size:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                cv2.putText(
                    frame, f"Latency: {latency_ms:.1f}ms", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                debug_info.caption(f"🔍 Debug: {debug_text}")

                if prediction:
                    nepali_char = get_nepali_character(prediction)
                    prediction_placeholder.success(f"**Prediction:** {nepali_char}")
                    confidence_placeholder.info(f"**Confidence:** {confidence:.2f}%")
                    latency_placeholder.metric("Latency", f"{latency_ms:.1f}ms")
                    with data_placeholder.expander("📊 Normalized Landmark Data"):
                        st.write(f"**Total Features:** {len(landmarks_data)}")
                        st.write(f"**Hand Size Factor:** {hand_size:.4f}")
                        landmark_cols = st.columns(3)
                        for i in range(0, len(landmarks_data), 3):
                            col_idx = (i % 9) // 3
                            with landmark_cols[col_idx]:
                                st.write(f"**LM {i // 3}:** X={landmarks_data[i]:.4f}")
                                st.write(f"Y={landmarks_data[i + 1]:.4f}")
                                st.write(f"Z={landmarks_data[i + 2]:.4f}")
                elif hand_detected:
                    prediction_placeholder.warning(
                        "❌ **No sign detected** — Confidence below threshold or invalid gesture"
                    )
                else:
                    prediction_placeholder.warning(
                        "❌ **No hand detected** — Position your hand clearly in the camera"
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            st.success("✅ Static detection stopped!")


# ═════════════════════════════════════════════════════════════
# DYNAMIC DETECTION MODE  —  ALL 7 BUGS FIXED
# ═════════════════════════════════════════════════════════════
elif dynamic_btn and gru_model is not None:
    st.info("🎬 **Dynamic Sign Language Detection** — Temporal sequence analysis")

    # ── CONFIG ──────────────────────────────────────────────
    # FIX BUG 7: derive feature length from the model's own input shape
    model_input_shape = gru_model.input_shape  # e.g. (None, 40, 360)
    expected_feature_len = model_input_shape[-1]          # actual feature dim
    sequence_length = model_input_shape[1]                # actual window size

    # Debug printout once — verify feature count matches your extractor
    st.sidebar.info(
        f"Model input: {model_input_shape}\n"
        f"Features expected: {expected_feature_len}\n"
        f"Sequence length: {sequence_length}"
    )

    min_sequence_frames = max(12, int(sequence_length * 0.4))
    confidence_threshold = 0.60

    # Temporal smoothing
    prediction_history_size = 5       # slightly wider window for stability
    ema_alpha = 0.4                    # slightly more responsive than 0.3
    temporal_consistency_threshold = 0.5

    # FIX BUG 2: hold last valid prediction across blank frames
    prediction_hold_frames = 20       # ~0.7 s at 30 fps

    st.write(
        f"Model window: **{sequence_length}** frames | "
        f"Fast trigger: **{min_sequence_frames}** frames | "
        f"Confidence threshold: **{confidence_threshold * 100:.0f}%** | "
        f"Features: **{expected_feature_len}**"
    )

    # ── OPEN WEBCAM ─────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access webcam!")
        st.stop()

    st.info("Make both hands visible with upper body. Press 'Q' to stop.")
    debug_info = st.empty()

    # ── STATE ────────────────────────────────────────────────
    landmark_sequence = deque(maxlen=sequence_length)
    frame_count = 0
    last_prediction = None
    last_confidence = 0.0             # stored as decimal 0–1 (FIX BUG 6)
    frames_since_last_valid = 0       # FIX BUG 2: hold-timer counter

    # Temporal smoothing buffers
    prediction_history = deque(maxlen=prediction_history_size)
    confidence_history = deque(maxlen=prediction_history_size)
    smoothed_confidence = -1.0        # FIX BUG 5: sentinel → no cold-start suppression

    # Latency / FPS tracking
    frame_times = deque(maxlen=30)
    prediction_latencies = deque(maxlen=30)
    landmark_extraction_latencies = deque(maxlen=30)

    # FIX BUG 4: declare ALL Streamlit metric placeholders OUTSIDE the loop
    metrics_placeholder = st.empty()

    # ── MAIN LOOP ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1
        frame_times.append(frame_start_time)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_results = hand_detector_dynamic.detect(mp_image)
        pose_result = pose_detector.detect(mp_image) if pose_detector else None

        hands = (
            hand_results.hand_landmarks
            if hand_results and hand_results.hand_landmarks
            else []
        )
        hands_detected = len(hands)
        pose_landmarks = (
            pose_result.pose_landmarks[0]
            if pose_result and pose_result.pose_landmarks
            else None
        )
        pose_detected = pose_landmarks is not None

        debug_text = (
            f"Hands: {hands_detected} | Pose: {'Yes' if pose_detected else 'No'} | "
            f"Buffer: {len(landmark_sequence)}/{sequence_length}"
        )

        # ── LANDMARK EXTRACTION (FIX BUG 1: gate relaxed to 1+ hand) ──
        hand_detected = hands_detected >= 1

        if hand_detected:
            for hand in hands[:2]:
                frame = draw_landmarks(frame, hand)
            if pose_landmarks:
                frame = draw_pose_skeleton(frame, pose_landmarks)

            landmark_extract_start = time.time()
            try:
                full_landmarks = []

                # Two hands: zero-pad missing second hand
                for idx in range(2):
                    if idx < hands_detected:
                        for lm in hands[idx]:
                            full_landmarks.extend([lm.x, lm.y, lm.z])
                    else:
                        full_landmarks.extend([0.0] * 63)

                # Pose: zero-pad if absent
                if pose_landmarks:
                    for lm in pose_landmarks:
                        full_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    # FIX BUG 7: pad to whatever the model actually expects
                    pose_feature_count = expected_feature_len - 126  # 2 hands = 126
                    full_landmarks.extend([0.0] * max(0, pose_feature_count))

                # Trim / pad to exact model feature length
                full_landmarks = full_landmarks[:expected_feature_len]
                full_landmarks.extend(
                    [0.0] * (expected_feature_len - len(full_landmarks))
                )

                landmark_sequence.append(
                    np.asarray(full_landmarks, dtype=np.float32).tolist()
                )
                landmark_extraction_latencies.append(
                    (time.time() - landmark_extract_start) * 1000
                )
                debug_text += f" | Features: {len(full_landmarks)}"

            except Exception as e:
                debug_text += f" | Extraction error: {e}"

        # ── INFERENCE (FIX BUG 3: stride removed — run every frame) ─────
        ran_inference = False
        if len(landmark_sequence) >= min_sequence_frames:
            try:
                model_sequence = resample_sequence_for_model(
                    list(landmark_sequence), sequence_length
                )
                sequence_array = np.expand_dims(model_sequence, axis=0)

                inference_start = time.time()
                raw_pred = gru_model.predict(sequence_array, verbose=0)
                prediction_latencies.append((time.time() - inference_start) * 1000)
                ran_inference = True

                prediction_class = int(np.argmax(raw_pred[0]))
                confidence = float(np.max(raw_pred[0]))

                # ── TEMPORAL SMOOTHING ──────────────────────────────────
                prediction_history.append(prediction_class)
                confidence_history.append(confidence)

                # FIX BUG 5: cold-start sentinel — skip EMA on first frame
                if smoothed_confidence < 0:
                    smoothed_confidence = confidence
                else:
                    smoothed_confidence = (
                        ema_alpha * confidence
                        + (1 - ema_alpha) * smoothed_confidence
                    )

                # Majority voting
                if len(prediction_history) >= 2:
                    try:
                        voted_prediction = mode(prediction_history)
                        vote_counts = {
                            p: list(prediction_history).count(p)
                            for p in set(prediction_history)
                        }
                        max_votes = max(vote_counts.values())
                        consistency = max_votes / len(prediction_history)
                    except StatisticsError:
                        voted_prediction = prediction_class
                        consistency = 1.0
                else:
                    voted_prediction = prediction_class
                    consistency = 1.0

                debug_text += (
                    f" | EMA conf: {smoothed_confidence * 100:.1f}% "
                    f"| Consistency: {consistency * 100:.0f}%"
                )

                # Accept if both EMA confidence and consistency pass
                if (
                    smoothed_confidence >= confidence_threshold
                    and consistency >= temporal_consistency_threshold
                ):
                    last_prediction = voted_prediction
                    last_confidence = smoothed_confidence   # FIX BUG 6: stay decimal
                    frames_since_last_valid = 0
                else:
                    frames_since_last_valid += 1

            except Exception as e:
                st.warning(f"Inference error: {e}")
                frames_since_last_valid += 1
        else:
            frames_since_last_valid += 1

        # FIX BUG 2: clear prediction only after hold window expires
        if frames_since_last_valid > prediction_hold_frames:
            last_prediction = None
            last_confidence = 0.0

        # ── FRAME ANNOTATIONS ───────────────────────────────────────────
        latency_ms = (time.time() - frame_start_time) * 1000

        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps = 0.0

        cv2.putText(
            frame,
            f"Buffer: {len(landmark_sequence)}/{sequence_length}  FPS: {fps:.1f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
        )

        if last_prediction is not None:
            class_name = get_gru_class_name(last_prediction, gru_config, REPO_ROOT)
            cv2.putText(
                frame,
                f"Sign: {class_name}  ({last_confidence * 100:.1f}%)",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )

        cv2.putText(
            frame,
            f"Latency: {latency_ms:.1f}ms",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
        )

        # ── STREAMLIT UI UPDATE ─────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)
        debug_info.caption(f"🔍 {debug_text}")

        # Prediction panel
        if last_prediction is not None:
            class_name = get_gru_class_name(last_prediction, gru_config, REPO_ROOT)
            prediction_placeholder.success(f"**🎯 Detected:** {class_name.upper()}")
            confidence_placeholder.info(
                f"**Confidence:** {last_confidence * 100:.1f}%"   # FIX BUG 6
            )
        elif len(landmark_sequence) >= min_sequence_frames:
            prediction_placeholder.warning(
                "⏳ Analysing — confidence too low or inconsistent"
            )
        else:
            prediction_placeholder.info(
                f"⏳ Collecting frames: {len(landmark_sequence)}/{min_sequence_frames}"
            )

        # Frame info bar
        if hand_detected:
            frame_info.info(
                f"✅ {hands_detected} hand(s) | Pose: {'yes' if pose_detected else 'no'} | "
                f"Buffer: {len(landmark_sequence)}/{sequence_length}"
            )
        else:
            frame_info.warning(
                f"❌ No hands detected | Buffer: {len(landmark_sequence)}/{sequence_length}"
            )

        # FIX BUG 4: update metrics via a single pre-declared placeholder
        avg_pred_lat = float(np.mean(prediction_latencies)) if prediction_latencies else 0.0
        avg_lm_lat = (
            float(np.mean(landmark_extraction_latencies))
            if landmark_extraction_latencies
            else 0.0
        )
        consistency_pct = 0.0
        if prediction_history:
            top_votes = max(
                list(prediction_history).count(p) for p in set(prediction_history)
            )
            consistency_pct = top_votes / len(prediction_history) * 100

        ema_display = (
            f"{smoothed_confidence * 100:.1f}%"
            if smoothed_confidence >= 0
            else "warming up…"
        )

        with metrics_placeholder.container():
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Frame latency", f"{latency_ms:.1f} ms")
            mc1.metric("Pred latency", f"{avg_pred_lat:.1f} ms")
            mc2.metric("FPS", f"{fps:.1f}")
            mc2.metric("LM latency", f"{avg_lm_lat:.1f} ms")
            mc3.metric("EMA confidence", ema_display)
            mc3.metric("Consistency", f"{consistency_pct:.0f}%")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    st.success("✅ Dynamic detection stopped!")


# ─────────────────────────────────────────────────────────────
# HELP TEXT
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
### 📖 How to use

#### 📷 Static detection
1. Click **Detect Static Sign Language**
2. Show one hand clearly to the camera
3. The app classifies each frame independently using the sklearn model
4. Press **Q** to stop

#### 🎬 Dynamic detection (fixed)
1. Click **Start Detecting Dynamic Language**
2. Show **one or both hands** — the model will zero-pad a missing second hand
3. Perform the sign over ~1–2 seconds while keeping your upper body visible
4. Prediction appears once the confidence + consistency thresholds are met
5. The last valid prediction is held for ~0.7 s after hands briefly leave frame
6. Press **Q** to stop

### ⚙️ What was fixed
| Bug | Fix |
|---|---|
| Hard two-hand + pose gate reset the buffer every frame | Gate relaxed to 1+ hand; buffer never cleared |
| `last_prediction` wiped on any imperfect frame | Hold timer keeps result for 20 frames |
| Inference stride silently halved detection chances | Stride removed; inference runs every frame |
| `st.columns` + `st.metric` inside the loop → crash | All metrics use a single pre-declared placeholder |
| EMA cold-start suppressed early predictions | Sentinel `-1` seeds EMA from first real confidence |
| Percentage vs decimal mismatch in threshold check | Confidence stored as decimal throughout |
| Hardcoded `expected_feature_len = 360` | Derived from `gru_model.input_shape` at runtime |
""")