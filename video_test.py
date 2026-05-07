import cv2
import numpy as np
import pickle
from collections import deque
import mediapipe as mp
import tensorflow as tf
import time

print("=" * 60)
print("FALL DETECTION TEST - CORRECTED")
print("=" * 60)

USE_WEBCAM = False   # True = webcam, False = video
VIDEO_PATH = "test_video.mp4"

model = tf.keras.models.load_model("fall_detection_model.h5")

with open("pose_data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = deque(maxlen=30)
pred_buffer = deque(maxlen=5)

FALL_THRESHOLD = 0.75
FALL_HOLD_SECONDS = 3
last_fall_time = 0


def angle3(a, b, c):
    a = np.array(a) - np.array(b)
    b = np.array(c) - np.array(b)

    cos_angle = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
    )

    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def extract_features_and_rules(landmarks):
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]
    l_knee = landmarks[25]
    r_knee = landmarks[26]
    l_ankle = landmarks[27]
    r_ankle = landmarks[28]
    head = landmarks[0]

    shoulder = np.array([
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_shoulder[1] + r_shoulder[1]) / 2
    ])

    hip = np.array([
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    ])

    ankle = np.array([
        (l_ankle[0] + r_ankle[0]) / 2,
        (l_ankle[1] + r_ankle[1]) / 2
    ])

    features = []

    # 1. Torso angle
    v = hip - shoulder
    vertical = np.array([0, 1])

    torso_angle = np.degrees(np.arccos(
        np.clip(np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6), -1, 1)
    ))
    features.append(torso_angle)

    # 2. Body height
    body_height = np.linalg.norm(ankle - np.array(head[:2]))
    features.append(body_height)

    # 3. Hip height
    hip_height = hip[1]
    features.append(hip_height)

    # 4. Aspect ratio
    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    aspect_ratio = np.clip(shoulder_width / (body_height + 1e-6), 0.1, 3.0)
    features.append(aspect_ratio)

    # 5–6. Leg angles
    left_leg_angle = angle3(l_hip, l_knee, l_ankle)
    right_leg_angle = angle3(r_hip, r_knee, r_ankle)

    features.append(left_leg_angle)
    features.append(right_leg_angle)

    # 7. Shoulder-ankle distance
    shoulder_ankle_distance = np.linalg.norm(shoulder - ankle)
    features.append(shoulder_ankle_distance)

    # 8–10 placeholders
    features.extend([0, 0, 0])

    # Rule-based posture check
    posture_looks_like_fall = (
        torso_angle > 45 or
        body_height < 0.45 or
        aspect_ratio > 0.75 or
        hip_height > 0.65
    )

    return np.array(features), posture_looks_like_fall, torso_angle, body_height, aspect_ratio


def pose_is_reliable(result):
    key_ids = [0, 11, 12, 23, 24, 25, 26, 27, 28]

    visibilities = [
        result.pose_landmarks.landmark[i].visibility
        for i in key_ids
    ]

    avg_visibility = np.mean(visibilities)

    return avg_visibility > 0.45


if USE_WEBCAM:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Running webcam... Press Q to quit")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Running video... Press Q to quit")

if not cap.isOpened():
    print("❌ Could not open camera/video")
    exit()


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    label = "Collecting..."
    color = (0, 255, 0)

    if result.pose_landmarks and pose_is_reliable(result):
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in result.pose_landmarks.landmark
        ])

        features, posture_fall, torso_angle, body_height, aspect_ratio = extract_features_and_rules(landmarks)

        features_scaled = scaler.transform([features])[0]
        sequence.append(features_scaled)

        if len(sequence) == 30:
            input_data = np.array(sequence).reshape(1, 30, 10)
            pred = model.predict(input_data, verbose=0)[0][0]

            pred_buffer.append(pred)
            avg_pred = float(np.mean(pred_buffer))

            # Final decision:
            # Model must say fall AND posture must also look like fall
            if avg_pred > FALL_THRESHOLD and posture_fall:
                last_fall_time = time.time()

            if time.time() - last_fall_time < FALL_HOLD_SECONDS:
                label = f"FALL ({avg_pred:.2f})"
                color = (0, 0, 255)
            else:
                label = f"Normal ({avg_pred:.2f})"
                color = (0, 255, 0)

            # Debug info
            cv2.putText(
                frame,
                f"angle:{torso_angle:.1f} height:{body_height:.2f} ratio:{aspect_ratio:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

    else:
        if time.time() - last_fall_time < FALL_HOLD_SECONDS:
            label = "FALL (pose lost)"
            color = (0, 0, 255)
        else:
            label = "Pose not reliable"
            color = (0, 255, 255)

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
pose.close()