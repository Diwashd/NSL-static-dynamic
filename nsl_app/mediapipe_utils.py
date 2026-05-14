import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def load_detectors(repo_root, num_hands=1):
    hand_task = os.path.join(repo_root, "hand_landmarker.task")
    pose_task = os.path.join(repo_root, "pose_landmarker.task")
    face_task = os.path.join(repo_root, "face_landmarker.task")

    hand_detector = None
    if os.path.exists(hand_task):
        try:
            base_options = python.BaseOptions(model_asset_path=hand_task)
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=num_hands)
            hand_detector = vision.HandLandmarker.create_from_options(options)
        except Exception:
            hand_detector = None

    pose_detector = None
    if os.path.exists(pose_task):
        try:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_task),
                running_mode=vision.RunningMode.IMAGE
            )
            pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        except Exception:
            pose_detector = None

    face_detector = None
    if os.path.exists(face_task):
        try:
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_task),
                running_mode=vision.RunningMode.IMAGE
            )
            face_detector = vision.FaceLandmarker.create_from_options(face_options)
        except Exception:
            face_detector = None

    return hand_detector, pose_detector, face_detector
