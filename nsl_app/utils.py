import math
import os
import numpy as np

POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FACE_INDICES = [33, 133, 159, 145, 174, 398, 362, 385, 387, 13]

NEPALI_MAPPING = {
    0: "क", 1: "ख", 2: "ग", 3: "घ", 4: "ङ",
    5: "च", 6: "छ", 7: "ज", 8: "झ", 9: "ञ",
    10: "ट", 11: "ठ", 12: "ड", 13: "ढ", 14: "ण",
    15: "त", 16: "थ", 17: "द", 18: "ध", 19: "न",
    20: "प", 21: "फ", 22: "ब", 23: "भ", 24: "म",
    25: "य", 26: "र", 27: "ल", 28: "व", 29: "श",
    30: "ष", 31: "स", 32: "ह", 33: "क्ष", 34: "त्र", 35: "ज्ञ"
}


def get_nepali_character(prediction_number):
    return NEPALI_MAPPING.get(int(prediction_number), str(prediction_number))


def get_dynamic_class_names(repo_root):
    data_dir = os.path.join(repo_root, "dataset_word", "landmarks", "final")
    if not os.path.isdir(data_dir):
        return None
    class_names = [
        name for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    return class_names if class_names else None


def get_gru_class_name(prediction_index, gru_config, repo_root=None):
    classes = None
    if gru_config and "classes" in gru_config:
        classes = gru_config["classes"]
    elif repo_root:
        classes = get_dynamic_class_names(repo_root)

    if classes and 0 <= prediction_index < len(classes):
        return classes[prediction_index]
    return str(prediction_index)


def extract_normalized_landmarks(hand):
    base_x = hand[0].x
    base_y = hand[0].y
    hand_size = math.sqrt((hand[17].x - hand[5].x) ** 2 + (hand[17].y - hand[5].y) ** 2)
    landmarks = []
    if hand_size > 0:
        for lm in hand:
            x = (lm.x - base_x) / hand_size
            y = (lm.y - base_y) / hand_size
            z = lm.z / hand_size
            landmarks.extend([x, y, z])
    return landmarks, hand_size


def extract_multimodal_landmarks(hand_detector, pose_detector, face_detector, mp_image, frame_shape):
    landmarks_list = []
    hand_size = 0
    try:
        hand_result = hand_detector.detect(mp_image)
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            base_x = hand[0].x
            base_y = hand[0].y
            hand_size = math.sqrt((hand[17].x - hand[5].x) ** 2 + (hand[17].y - hand[5].y) ** 2)
            hand_landmarks = []
            if hand_size > 0:
                for lm in hand:
                    x = (lm.x - base_x) / hand_size
                    y = (lm.y - base_y) / hand_size
                    z = lm.z / hand_size
                    hand_landmarks.extend([x, y, z])
            landmarks_list.append(np.array(hand_landmarks))
        else:
            landmarks_list.append(np.zeros(63))
    except Exception:
        landmarks_list.append(np.zeros(63))

    try:
        if pose_detector:
            pose_result = pose_detector.detect(mp_image)
            if pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]
                pose_features = []
                for idx in POSE_INDICES:
                    if idx < len(pose_landmarks):
                        lm = pose_landmarks[idx]
                        pose_features.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(np.array(pose_features))
            else:
                landmarks_list.append(np.zeros(39))
        else:
            landmarks_list.append(np.zeros(39))
    except Exception:
        landmarks_list.append(np.zeros(39))

    try:
        if face_detector:
            face_result = face_detector.detect(mp_image)
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]
                face_features = []
                for idx in FACE_INDICES:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        face_features.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(np.array(face_features))
            else:
                landmarks_list.append(np.zeros(30))
        else:
            landmarks_list.append(np.zeros(30))
    except Exception:
        landmarks_list.append(np.zeros(30))

    multimodal_landmarks = np.concatenate(landmarks_list)
    return multimodal_landmarks, hand_size
