import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import pandas as pd
from tqdm import tqdm
import urllib.request
import math

# -----------------------------------------------------
# STEP 1: Download MediaPipe Model
# -----------------------------------------------------

model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded.")

# -----------------------------------------------------
# STEP 2: Initialize MediaPipe
# -----------------------------------------------------

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------------------------------
# STEP 3: Dataset Path
# -----------------------------------------------------

dataset_path = "dataset"
data = []

# -----------------------------------------------------
# STEP 4: Loop Through Dataset
# -----------------------------------------------------

for bg in ["Plain Background", "Random Background"]:

    bg_path = os.path.join(dataset_path, bg)

    if not os.path.exists(bg_path):
        continue

    for label in os.listdir(bg_path):

        class_path = os.path.join(bg_path, label)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)

        for img in tqdm(images, desc=f"{bg}-{label}"):

            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_path, img)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            results = detector.detect(mp_image)

            if results.hand_landmarks:

                hand = results.hand_landmarks[0]

                # -------------------------------------------------
                # STEP 5: Detect Left/Right Hand
                # -------------------------------------------------
                is_left = False
                if results.handedness:
                    is_left = results.handedness[0][0].category_name == "Left"

                # -------------------------------------------------
                # STEP 6: Translation (Wrist as Origin)
                # -------------------------------------------------
                base_x = hand[0].x
                base_y = hand[0].y

                # -------------------------------------------------
                # STEP 7: Scale Normalization
                # -------------------------------------------------
                hand_size = math.sqrt(
                    (hand[17].x - hand[5].x) ** 2 +
                    (hand[17].y - hand[5].y) ** 2
                )

                if hand_size == 0:
                    continue

                # -------------------------------------------------
                # STEP 8: Rotation Normalization
                # Align wrist → middle finger
                # -------------------------------------------------
                dx = hand[9].x - hand[0].x
                dy = hand[9].y - hand[0].y

                angle = math.atan2(dy, dx)

                landmarks = []

                for lm in hand:

                    # Translate
                    x = lm.x - base_x
                    y = lm.y - base_y

                    # -------------------------------------------------
                    # FIX 1: Mirror Left Hand
                    # -------------------------------------------------
                    if is_left:
                        x = -x

                    # -------------------------------------------------
                    # FIX 2: Rotate to Standard Orientation
                    # -------------------------------------------------
                    x_rot = x * math.cos(-angle) - y * math.sin(-angle)
                    y_rot = x * math.sin(-angle) + y * math.cos(-angle)

                    # -------------------------------------------------
                    # Normalize Scale
                    # -------------------------------------------------
                    x_rot /= hand_size
                    y_rot /= hand_size
                    z = lm.z / hand_size

                    landmarks.extend([x_rot, y_rot, z])

                # Append label + background
                landmarks.append(int(label))
                landmarks.append(bg)

                data.append(landmarks)

# -----------------------------------------------------
# STEP 9: Save Dataset
# -----------------------------------------------------

print("Creating dataframe...")

columns = []

for i in range(21):
    columns.append(f"x{i}")
    columns.append(f"y{i}")
    columns.append(f"z{i}")

columns.append("label")
columns.append("bg_type")  # NEW

df = pd.DataFrame(data, columns=columns)

df.to_csv("nsl_landmarks_v2.csv", index=False)

print("Landmark extraction complete.")
print("Saved as: nsl_landmarks_v2.csv")
print("Total samples:", len(df))