import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

POSE_DRAW_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def draw_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for connection in HAND_CONNECTIONS:
        start, end = connection
        if start < len(hand_landmarks) and end < len(hand_landmarks):
            x1 = int(hand_landmarks[start].x * w)
            y1 = int(hand_landmarks[start].y * h)
            x2 = int(hand_landmarks[end].x * w)
            y2 = int(hand_landmarks[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_pose_skeleton(image, pose_landmarks):
    h, w, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for start, end in POSE_CONNECTIONS:
        if start < len(pose_landmarks) and end < len(pose_landmarks):
            x1 = int(pose_landmarks[start].x * w)
            y1 = int(pose_landmarks[start].y * h)
            x2 = int(pose_landmarks[end].x * w)
            y2 = int(pose_landmarks[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 165, 0), 2)

    for idx in POSE_DRAW_INDICES:
        if idx < len(pose_landmarks):
            x = int(pose_landmarks[idx].x * w)
            y = int(pose_landmarks[idx].y * h)
            cv2.circle(image, (x, y), 4, (0, 255, 255), -1)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
