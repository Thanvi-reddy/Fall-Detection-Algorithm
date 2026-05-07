import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import pickle

# Load model
model = tf.keras.models.load_model("fall_detection_model.h5")
scaler = pickle.load(open("pose_data/scaler.pkl", 'rb'))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = deque(maxlen=30)

prev_hip = None
prev_shoulder = None


def extract_features(landmarks, prev_hip, prev_shoulder):
    head = landmarks[0]
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]
    l_knee = landmarks[25]
    r_knee = landmarks[26]
    l_ankle = landmarks[27]
    r_ankle = landmarks[28]

    features = []

    shoulder = np.array([(l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2])
    hip = np.array([(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2])

    v = hip - shoulder
    torso = np.degrees(np.arccos(np.clip(np.dot(v, [0,1])/(np.linalg.norm(v)+1e-6), -1, 1)))
    features.append(torso)

    height = np.linalg.norm(np.array(head[:2]) - np.array([(l_ankle[0]+r_ankle[0])/2, (l_ankle[1]+r_ankle[1])/2]))
    features.append(height)

    hip_h = hip[1]
    features.append(hip_h)

    width = abs(l_sh[0] - r_sh[0])
    features.append(width/(height+1e-6))

    def angle(a,b,c):
        a,b,c = np.array(a),np.array(b),np.array(c)
        ba, bc = a-b, c-b
        return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6), -1,1)))

    features.append(angle(l_hip,l_knee,l_ankle))
    features.append(angle(r_hip,r_knee,r_ankle))

    ankle = np.array([(l_ankle[0]+r_ankle[0])/2, (l_ankle[1]+r_ankle[1])/2])
    features.append(np.linalg.norm(shoulder-ankle))

    hip_vel = hip_h - prev_hip if prev_hip is not None else 0
    features.append(hip_vel)

    sh_h = shoulder[1]
    sh_vel = sh_h - prev_shoulder if prev_shoulder is not None else 0
    features.append(sh_vel)

    features.append(np.linalg.norm(shoulder - np.array(head[:2])))

    return np.array(features), hip_h, sh_h


print("Running detection... Press Q to quit")

confirm = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        pts = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])

        feats, prev_hip, prev_shoulder = extract_features(pts, prev_hip, prev_shoulder)
        feats = scaler.transform(feats.reshape(1,-1))[0]

        sequence.append(feats)

        if len(sequence) == 30:
            pred = model.predict(np.array(sequence).reshape(1,30,10), verbose=0)[0][0]

            if pred > 0.6:
                confirm += 1
            else:
                confirm = 0

            if confirm > 15:
                cv2.putText(frame, "FALL DETECTED!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            else:
                cv2.putText(frame, "Normal", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()