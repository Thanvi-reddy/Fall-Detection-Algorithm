import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("Loading aggregated poses...")

# Use aggregated dataset
INPUT_FILE = "pose_data/poses_agg.pkl"

with open(INPUT_FILE, "rb") as f:
    all_poses = pickle.load(f)

print(f"Loaded {len(all_poses)} samples")

features_list = []
labels_list = []

for item in all_poses:
    features = item["features"]
    label = item["label"]

    features_list.append(features)
    labels_list.append(1 if label == "fall" else 0)

X = np.array(features_list)
y = np.array(labels_list)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("\nNormalizing features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save normalized aggregated features
np.save("pose_data/features.npy", X_scaled)
np.save("pose_data/labels.npy", y)

with open("pose_data/scaler_agg.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Saved:")
print("features.npy")
print("labels.npy")
print("scaler_agg.pkl")

print("\nClass distribution:")
print("Normal:", np.sum(y == 0))
print("Fall:", np.sum(y == 1))