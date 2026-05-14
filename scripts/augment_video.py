import os
import cv2
import numpy as np

INPUT_DIR = "dataset_word/raw_videos"
OUTPUT_DIR = "dataset_word/augmented_videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Noise function
# -----------------------------
def add_noise(frame):
    noise = np.random.normal(0, 5, frame.shape).astype(np.float32)
    frame = frame.astype(np.float32) + noise
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    alpha = 1.0 + np.random.uniform(-0.1, 0.1)
    beta = np.random.uniform(-10, 10)

    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if np.random.rand() < 0.3:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame


# -----------------------------
# Save video
# -----------------------------
def save_video(frames, path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for f in frames:
        out.write(f)

    out.release()


# -----------------------------
# Augment one video
# -----------------------------
def augment_video(video_path, output_dir, base_name):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate if portrait (height > width)
        h, w = frame.shape[:2]
        if h > w:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frames.append(frame)

    cap.release()

    if len(frames) < 5:
        print(f"⚠️ Skipped short video: {video_path}")
        return

    h, w = frames[0].shape[:2]

    print(f"     Frames: {len(frames)} | FPS: {fps}")

    # 1. original
    save_video(frames, os.path.join(output_dir, f"{base_name}_orig.mp4"), fps, w, h)

    # 2. slow
    slow = []
    for f in frames:
        slow.append(f)
        slow.append(f)
    save_video(slow, os.path.join(output_dir, f"{base_name}_slow.mp4"), fps, w, h)

    # 3. fast
    fast = frames[::2]
    save_video(fast, os.path.join(output_dir, f"{base_name}_fast.mp4"), fps, w, h)

    # 4. flip
    flip = [cv2.flip(f, 1) for f in frames]
    save_video(flip, os.path.join(output_dir, f"{base_name}_flip.mp4"), fps, w, h)

    # 5. crop
    crop = [f[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)] for f in frames]
    crop = [cv2.resize(f, (w, h)) for f in crop]
    save_video(crop, os.path.join(output_dir, f"{base_name}_crop.mp4"), fps, w, h)

    # 6. noise
    noise = [add_noise(f) for f in frames]
    save_video(noise, os.path.join(output_dir, f"{base_name}_noise.mp4"), fps, w, h)


# -----------------------------
# MAIN LOOP
# -----------------------------
print("📁 INPUT DIR:", os.path.abspath(INPUT_DIR))

classes = os.listdir(INPUT_DIR)
print("📂 Classes found:", classes)

for cls in classes:
    class_path = os.path.join(INPUT_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(output_class_path, exist_ok=True)

    videos = os.listdir(class_path)

    print(f"\n📦 Processing class: {cls} ({len(videos)} files)")

    for vid in videos:
        print(f"   Found file: {vid}")

        # Support multiple formats
        if not vid.lower().endswith((".mp4", ".mov", ".avi")):
            print("     ⏭ Skipped (unsupported format)")
            continue

        video_path = os.path.join(class_path, vid)

        # ✅ FIX: include class name in output
        base_name = f"{cls}_{os.path.splitext(vid)[0]}"

        print(f"   → Processing: {vid}")

        augment_video(video_path, output_class_path, base_name)

print("\n✅ Augmentation COMPLETE")