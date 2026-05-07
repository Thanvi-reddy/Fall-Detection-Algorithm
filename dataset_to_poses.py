import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from pathlib import Path

print("=" * 60)
print("DATASET → POSE FEATURES CONVERSION")
print("=" * 60)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs("pose_data", exist_ok=True)

DATASET_PATH = "datasets"


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

    features = []

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

    # 1. Torso angle
    v = hip - shoulder
    vertical = np.array([0, 1])

    torso_angle = np.degrees(
        np.arccos(
            np.clip(
                np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6),
                -1,
                1
            )
        )
    )
    features.append(torso_angle)

    # 2. Body height
    body_height = np.linalg.norm(ankle - np.array(head[:2]))
    features.append(body_height)

    # 3. Hip height
    features.append(hip[1])

    # 4. Aspect ratio
    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    aspect_ratio = shoulder_width / (body_height + 1e-6)
    aspect_ratio = np.clip(aspect_ratio, 0.1, 3.0)
    features.append(aspect_ratio)

    # 5–6. Leg angles
    def angle3(a, b, c):
        a = np.array(a) - np.array(b)
        c = np.array(c) - np.array(b)

        cos_angle = np.dot(a, c) / (
            np.linalg.norm(a) * np.linalg.norm(c) + 1e-6
        )

        return np.degrees(
            np.arccos(np.clip(cos_angle, -1, 1))
        )

    features.append(angle3(l_hip, l_knee, l_ankle))
    features.append(angle3(r_hip, r_knee, r_ankle))

    # 7. Shoulder-to-ankle distance
    features.append(np.linalg.norm(shoulder - ankle))

    # 8–10. Placeholder velocity features
    # These are kept as 0 because dataset images are processed frame-by-frame.
    features.extend([0, 0, 0])

    return np.array(features)


def process_image(img_path):
    img = cv2.imread(str(img_path))

    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    landmarks = np.array([
        [lm.x, lm.y, lm.z]
        for lm in result.pose_landmarks.landmark
    ])

    return extract_features(landmarks)


all_data = []

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset folder not found: {DATASET_PATH}")
    print("Create folder: D:\\fall_detection_project\\datasets")
    exit()

folders = sorted(os.listdir(DATASET_PATH))

for folder in folders:
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    folder_lower = folder.lower()

    if "fall" in folder_lower:
        label = "fall"
    elif "adl" in folder_lower:
        label = "normal"
    else:
        continue

    print(f"\nProcessing {folder} ({label})")

    images = (
        list(Path(folder_path).glob("*.png")) +
        list(Path(folder_path).glob("*.jpg")) +
        list(Path(folder_path).glob("*.jpeg"))
    )

    images = sorted(images)

    success = 0
    failed = 0

    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(images)}")

        features = process_image(img_path)

        if features is not None:
            all_data.append({
                "features": features,
                "label": label,
                "source": str(img_path)
            })
            success += 1
        else:
            failed += 1

    print(f"  Extracted: {success}/{len(images)}")
    print(f"  Failed: {failed}")

with open("pose_data/poses.pkl", "wb") as f:
    pickle.dump(all_data, f)

pose.close()

print("\n" + "=" * 60)
print("DONE")
print(f"Total frames: {len(all_data)}")
print(f"Fall frames: {sum(1 for x in all_data if x['label'] == 'fall')}")
print(f"Normal frames: {sum(1 for x in all_data if x['label'] == 'normal')}")
print("Saved: pose_data/poses.pkl")
print("=" * 60)