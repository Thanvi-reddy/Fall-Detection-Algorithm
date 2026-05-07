import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading pose data...")

with open("pose_data/poses.pkl", 'rb') as f:
    all_poses = pickle.load(f)

features_list = []
labels_list = []

for pose_data in all_poses:
    features = pose_data['features']
    label = pose_data['label']
    
    features_list.append(features)
    labels_list.append(1 if label == 'fall' else 0)

X = np.array(features_list)
y = np.array(labels_list)

print(f"Total samples: {len(X)}")
print(f"Features per frame: {X.shape[1]}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save
np.save("pose_data/features.npy", X_scaled)
np.save("pose_data/labels.npy", y)

with open("pose_data/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print("✓ Features saved")