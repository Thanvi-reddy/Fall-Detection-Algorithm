import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from datetime import datetime

print("Initializing MediaPipe...")

# ✅ Correct MediaPipe initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

print("✓ MediaPipe loaded successfully\n")

# Create output directory
os.makedirs("pose_data", exist_ok=True)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = max(np.linalg.norm(ba) * np.linalg.norm(bc), 1e-6)
    cosine_angle = np.dot(ba, bc) / denominator

    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def get_torso_angle_corrected(landmarks):
    shoulder = np.array([landmarks[11][0], landmarks[11][1]])
    hip = np.array([landmarks[23][0], landmarks[23][1]])

    v = hip - shoulder
    vertical = np.array([0, 1])

    denominator = max(np.linalg.norm(v), 1e-6)
    cos_angle = np.dot(v, vertical) / denominator

    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def get_body_height(landmarks):
    head = np.array([landmarks[0][0], landmarks[0][1]])
    ankle_left = np.array([landmarks[27][0], landmarks[27][1]])
    ankle_right = np.array([landmarks[28][0], landmarks[28][1]])

    ankle = (ankle_left + ankle_right) / 2
    return np.linalg.norm(ankle - head)


def extract_features(pose_landmarks, prev_hip=None, prev_shoulder=None):
    head = pose_landmarks[0]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]
    left_knee = pose_landmarks[25]
    right_knee = pose_landmarks[26]
    left_ankle = pose_landmarks[27]
    right_ankle = pose_landmarks[28]

    features = []

    # 1. Torso angle
    torso_angle = get_torso_angle_corrected(pose_landmarks)
    features.append(torso_angle)

    # 2. Body height
    body_height = get_body_height(pose_landmarks)
    features.append(body_height)

    # 3. Hip height
    hip_center = (np.array(left_hip) + np.array(right_hip)) / 2
    hip_height = hip_center[1]
    features.append(hip_height)

    # 4. Aspect ratio
    shoulder_center = (np.array(left_shoulder) + np.array(right_shoulder)) / 2
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

    if body_height < 0.1:
        aspect_ratio = 0.5
    else:
        aspect_ratio = np.clip(shoulder_width / body_height, 0.1, 3.0)

    features.append(aspect_ratio)

    # 5. Left leg angle
    features.append(calculate_angle(left_hip, left_knee, left_ankle))

    # 6. Right leg angle
    features.append(calculate_angle(right_hip, right_knee, right_ankle))

    # 7. Shoulder-ankle distance
    ankle_center = (np.array(left_ankle) + np.array(right_ankle)) / 2
    distance = np.linalg.norm(shoulder_center - ankle_center)
    features.append(distance)

    # 8. Hip velocity
    hip_velocity = hip_height - prev_hip if prev_hip is not None else 0
    features.append(hip_velocity)

    # 9. Shoulder velocity
    shoulder_height = shoulder_center[1]
    shoulder_velocity = shoulder_height - prev_shoulder if prev_shoulder is not None else 0
    features.append(shoulder_velocity)

    # 10. Movement magnitude
    movement = np.linalg.norm(shoulder_center - np.array(head))
    features.append(movement)

    return np.array(features), hip_height, shoulder_height


def collect_posture_data(label, duration=30):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not working!")
        return []

    poses = []
    prev_hip = None
    prev_shoulder = None

    print(f"\nRecording {label.upper()} for {duration} seconds...")

    start_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])

            features, prev_hip, prev_shoulder = extract_features(
                landmarks, prev_hip, prev_shoulder
            )

            poses.append({
                "features": features,
                "label": label
            })

        elapsed = (datetime.now() - start_time).total_seconds()

        cv2.putText(frame, f"{label} | {duration - elapsed:.1f}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)

        if elapsed >= duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return poses


# ===== MAIN =====
all_data = []

print("Collect NORMAL data")
all_data += collect_posture_data("normal", 20)

print("Collect FALL data")
all_data += collect_posture_data("fall", 20)

with open("pose_data/poses.pkl", "wb") as f:
    pickle.dump(all_data, f)

print("✅ Data saved successfully!")