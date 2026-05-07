import numpy as np

X = np.load("pose_data/features.npy")
y = np.load("pose_data/labels.npy")

SEQUENCE_LENGTH = 30
STEP = 15

X_seq = []
y_seq = []

for i in range(0, len(X) - SEQUENCE_LENGTH, STEP):
    sequence = X[i:i + SEQUENCE_LENGTH]
    
    # ✅ IMPORTANT FIX
    label = 1 if np.max(y[i:i + SEQUENCE_LENGTH]) == 1 else 0
    
    X_seq.append(sequence)
    y_seq.append(label)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Shuffle
idx = np.random.permutation(len(X_seq))
X_seq = X_seq[idx]
y_seq = y_seq[idx]

np.save("pose_data/X_sequences.npy", X_seq)
np.save("pose_data/y_sequences.npy", y_seq)

print(f"Sequences: {len(X_seq)}")
print("✓ Saved sequences")