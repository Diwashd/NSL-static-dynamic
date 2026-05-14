"""
=========================================================
ADVANCED LIGHTWEIGHT NSL LANDMARK EXTRACTION PIPELINE
(OPTIMIZED FOR REAL-TIME + LOW LATENCY)

✔ Hand + Pose only
✔ Face REMOVED (noise reduction)
✔ Relative normalization
✔ Scale normalization
✔ Velocity features
✔ Acceleration features
✔ Bone vector features
✔ Geometric features
✔ Temporal interpolation
✔ Fixed-length stable sequences
✔ Thesis-grade pipeline
=========================================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import urllib.request

from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "dataset_word/augmented_videos"
OUTPUT_DIR = "dataset_word/landmarks/final"

SEQUENCE_LENGTH = 40

NUM_HAND = 21
NUM_POSE = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DOWNLOAD MODELS
# =========================================================

MODELS = [
    (
        "hand.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    (
        "pose.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
]

for path, url in MODELS:

    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)


# =========================================================
# MEDIAPIPE INIT
# =========================================================

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="hand.task"
        ),
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
)

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="pose.task"
        ),
        running_mode=vision.RunningMode.IMAGE
    )
)


# =========================================================
# HAND CONNECTIONS
# =========================================================

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]


# =========================================================
# TEMPORAL MEMORY
# =========================================================

prev_hand = None
prev_pose = None

prev_hand_vel = None
prev_pose_vel = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def wrist_normalize(points, wrist_idx=0):

    return points - points[wrist_idx]


def scale_normalize(points):

    scale = np.linalg.norm(points[0] - points[9]) + 1e-6
    return points / scale


def compute_angle(a, b, c):

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        (np.linalg.norm(ba) + 1e-6) *
        (np.linalg.norm(bc) + 1e-6)
    )

    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return angle


def compute_bones(hand_points):

    bones = []

    for a, b in HAND_CONNECTIONS:

        bone = hand_points[b] - hand_points[a]
        bones.append(bone)

    return np.array(bones).flatten()


def temporal_resize(sequence, target_len=40):

    old_len = len(sequence)

    if old_len == target_len:
        return sequence

    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, target_len)

    resized = []

    for i in range(sequence.shape[1]):

        interp = np.interp(
            new_idx,
            old_idx,
            sequence[:, i]
        )

        resized.append(interp)

    resized = np.array(resized).T

    return resized


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(frame):

    global prev_hand
    global prev_pose

    global prev_hand_vel
    global prev_pose_vel

    frame = cv2.resize(frame, (960, 540))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # =====================================================
    # HAND LANDMARKS
    # =====================================================

    try:

        result = hand_detector.detect(mp_image)

        if result.hand_landmarks:

            hand = np.array([
                [p.x, p.y, p.z]
                for p in result.hand_landmarks[0]
            ])

        else:
            hand = np.zeros((NUM_HAND, 3))

    except:

        hand = np.zeros((NUM_HAND, 3))

    hand = wrist_normalize(hand)
    hand = scale_normalize(hand)

    # =====================================================
    # HAND VELOCITY
    # =====================================================

    if prev_hand is not None:
        hand_vel = hand - prev_hand
    else:
        hand_vel = np.zeros_like(hand)

    # =====================================================
    # HAND ACCELERATION
    # =====================================================

    if prev_hand_vel is not None:
        hand_acc = hand_vel - prev_hand_vel
    else:
        hand_acc = np.zeros_like(hand)

    prev_hand = hand.copy()
    prev_hand_vel = hand_vel.copy()

    # =====================================================
    # BONE FEATURES
    # =====================================================

    hand_bones = compute_bones(hand)

    # =====================================================
    # POSE LANDMARKS
    # =====================================================

    try:

        result = pose_detector.detect(mp_image)

        if result.pose_landmarks:

            lm = result.pose_landmarks[0]

            idx = [
                11,12,
                13,14,
                15,16,
                23,24,
                25,26,
                27,28
            ]

            pose = np.array([
                [lm[i].x, lm[i].y, lm[i].z]
                for i in idx
            ])

        else:
            pose = np.zeros((NUM_POSE, 3))

    except:

        pose = np.zeros((NUM_POSE, 3))

    pose = wrist_normalize(pose)

    # =====================================================
    # POSE VELOCITY
    # =====================================================

    if prev_pose is not None:
        pose_vel = pose - prev_pose
    else:
        pose_vel = np.zeros_like(pose)

    # =====================================================
    # POSE ACCELERATION
    # =====================================================

    if prev_pose_vel is not None:
        pose_acc = pose_vel - prev_pose_vel
    else:
        pose_acc = np.zeros_like(pose)

    prev_pose = pose.copy()
    prev_pose_vel = pose_vel.copy()

    # =====================================================
    # GEOMETRIC FEATURES
    # =====================================================

    try:

        elbow_angle_left = compute_angle(
            pose[0],
            pose[2],
            pose[4]
        )

        elbow_angle_right = compute_angle(
            pose[1],
            pose[3],
            pose[5]
        )

        wrist_distance = np.linalg.norm(
            hand[0] - hand[9]
        )

    except:

        elbow_angle_left = 0
        elbow_angle_right = 0
        wrist_distance = 0

    # =====================================================
    # FINAL FEATURE VECTOR
    # =====================================================

    features = np.concatenate([

        # landmarks
        hand.flatten(),
        pose.flatten(),

        # velocity
        hand_vel.flatten(),
        pose_vel.flatten(),

        # acceleration
        hand_acc.flatten(),
        pose_acc.flatten(),

        # bones
        hand_bones,

        # geometry
        [elbow_angle_left],
        [elbow_angle_right],
        [wrist_distance]

    ])

    return features.astype(np.float32)


# =========================================================
# VIDEO TO TEMPORAL SEQUENCE
# =========================================================

def video_to_sequence(video_path):

    global prev_hand
    global prev_pose

    global prev_hand_vel
    global prev_pose_vel

    prev_hand = None
    prev_pose = None

    prev_hand_vel = None
    prev_pose_vel = None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return np.zeros((SEQUENCE_LENGTH, 363))

    sequence = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        feat = extract_features(frame)

        sequence.append(feat)

    cap.release()

    if len(sequence) == 0:

        return np.zeros((SEQUENCE_LENGTH, 363))

    sequence = np.array(sequence)

    # temporal interpolation
    sequence = temporal_resize(
        sequence,
        SEQUENCE_LENGTH
    )

    return sequence


# =========================================================
# MAIN PIPELINE
# =========================================================

if not os.path.exists(INPUT_DIR):
    print(f"ERROR: input directory not found: {INPUT_DIR}")
    print("Create the directory and add class subfolders with .mp4 videos, or set INPUT_DIR correctly.")
    raise SystemExit(1)

classes = sorted(os.listdir(INPUT_DIR))

print("\n=================================================")
print("NSL LANDMARK EXTRACTION STARTED")
print("=================================================")

for cls in classes:

    input_path = os.path.join(INPUT_DIR, cls)
    output_path = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(output_path, exist_ok=True)

    videos = [
        v for v in os.listdir(input_path)
        if v.endswith(".mp4")
    ]

    print(f"\nClass: {cls}")
    print(f"Videos: {len(videos)}")

    for video in tqdm(videos):

        video_path = os.path.join(input_path, video)

        sequence = video_to_sequence(video_path)

        save_name = video.replace(".mp4", ".npy")

        np.save(
            os.path.join(output_path, save_name),
            sequence
        )

print("\n=================================================")
print("LANDMARK EXTRACTION COMPLETE")
print("=================================================")
print("Saved to:", OUTPUT_DIR)