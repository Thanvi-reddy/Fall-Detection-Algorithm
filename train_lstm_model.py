import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

print("Loading aggregated sequences...")

X = np.load("pose_data/X_sequences.npy")
y = np.load("pose_data/y_sequences.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nTrain:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 40)),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting aggregated model training...\n")

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8
)

loss, acc = model.evaluate(X_test, y_test)

print(f"\nAggregated Test Accuracy: {acc * 100:.2f}%")

model.save("fall_detection_model_agg.h5")
print("✓ Aggregated model saved as fall_detection_model_agg.h5")