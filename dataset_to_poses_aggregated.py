import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from collections import deque
import mediapipe as mp

print("=" * 60)
print("AGGREGATED DATASET TO POSES")
print("=" * 60)

DATASET_PATH = "datasets"
OUTPUT_FILE = "pose_data/poses_agg.pkl"

WINDOW_SIZE = 5

os.makedirs("pose_data", exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_features(landmarks):
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

    v = hip - shoulder
    vertical = np.array([0, 1])
    torso_angle = np.degrees(np.arccos(
        np.clip(np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6), -1, 1)
    ))
    features.append(torso_angle)

    body_height = np.linalg.norm(ankle - np.array(head[:2]))
    features.append(body_height)

    features.append(hip[1])

    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    features.append(np.clip(shoulder_width / (body_height + 1e-6), 0.1, 3.0))

    def angle3(a, b, c):
        a = np.array(a) - np.array(b)
        b = np.array(c) - np.array(b)
        return np.degrees(np.arccos(
            np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6), -1, 1)
        ))

    features.append(angle3(l_hip, l_knee, l_ankle))
    features.append(angle3(r_hip, r_knee, r_ankle))

    features.append(np.linalg.norm(shoulder - ankle))

    features.extend([0, 0, 0])

    return np.array(features)


def aggregate_window(window):
    arr = np.array(window)

    agg = []
    agg.extend(np.mean(arr, axis=0))
    agg.extend(np.std(arr, axis=0))
    agg.extend(np.min(arr, axis=0))
    agg.extend(np.max(arr, axis=0))

    return np.array(agg)


def process_folder(folder_path, label):
    images = (
        list(Path(folder_path).glob("*.png")) +
        list(Path(folder_path).glob("*.jpg")) +
        list(Path(folder_path).glob("*.jpeg"))
    )

    images = sorted(images)

    window = deque(maxlen=WINDOW_SIZE)
    output = []

    success = 0
    failed = 0

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))

        if img is None:
            failed += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            failed += 1
            continue

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in result.pose_landmarks.landmark
        ])

        features = extract_features(landmarks)
        window.append(features)
        success += 1

        if len(window) == WINDOW_SIZE:
            agg_features = aggregate_window(window)

            output.append({
                "features": agg_features,
                "label": label,
                "source": str(img_path)
            })

        if i % 100 == 0:
            print(f"  Progress {i}/{len(images)}")

    print(f"  Success: {success}, Failed: {failed}, Aggregated: {len(output)}")
    return output


all_data = []

for folder in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    if "fall" in folder.lower():
        label = "fall"
    elif "adl" in folder.lower():
        label = "normal"
    else:
        continue

    print(f"\nProcessing {folder} -> {label}")
    all_data.extend(process_folder(folder_path, label))

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(all_data, f)

pose.close()

print("\nDONE")
print("Saved:", OUTPUT_FILE)
print("Total samples:", len(all_data))
print("Fall:", sum(1 for x in all_data if x["label"] == "fall"))
print("Normal:", sum(1 for x in all_data if x["label"] == "normal"))