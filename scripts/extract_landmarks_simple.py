import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import pandas as pd
from tqdm import tqdm
import urllib.request

# Download model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)

# Create options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

dataset_path = "dataset"
data = []

for bg in ["Plain Background", "Random Background"]:
    bg_path = os.path.join(dataset_path, bg)
    for label in os.listdir(bg_path):
        class_path = os.path.join(bg_path, label)
        for img in tqdm(os.listdir(class_path), desc=f"{bg}-{label}"):
            img_path = os.path.join(class_path, img)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = detector.detect(mp_image)
            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                base_x = hand[0].x
                base_y = hand[0].y
                landmarks = []
                for lm in hand:
                    landmarks.append(lm.x - base_x)
                    landmarks.append(lm.y - base_y)
                    landmarks.append(lm.z)
                landmarks.append(int(label))
                data.append(landmarks)

df = pd.DataFrame(data)
df.to_csv("nsl_landmarks.csv", index=False)

print("Extraction complete")