# 🚨 Posture-Based Fall Detection System

<h3 align="center">
Real-Time Fall Detection using Computer Vision and Deep Learning
</h3>

---

## 📌 Overview

This project is a real-time posture-based fall detection system developed using Computer Vision and Deep Learning techniques.

The system analyzes human posture and movement patterns from video streams to detect abnormal fall events in real time. It supports both single-person and multi-person detection workflows for healthcare monitoring and smart surveillance applications.

The project includes pose extraction, feature engineering, sequence generation, LSTM-based training, and real-time fall event prediction pipelines.

---

## 🚀 Features

- 🎥 Real-time fall detection
- 🧍 Single-person posture detection
- 👥 Multi-person fall detection
- 🤖 AI-based posture analysis
- ⚡ Real-time monitoring
- 🚨 Fall alert prediction
- 🧠 Deep learning workflow

---

## 🧰 Tech Stack

- Python
- OpenCV
- TensorFlow
- LSTM
- Computer Vision
- Deep Learning

---

## 📂 Project Structure

```text
Fall-Detection-Algorithm/
├── collect_live_data.py
├── create_sequences.py
├── dataset_to_poses.py
├── dataset_to_poses_aggregated.py
├── feature_extraction.py
├── normalize_features.py
├── real_time_detection.py
├── train_lstm_model.py
├── train_model.py
├── video_test.py
├── fall_detection_model.h5
├── fall_detection_model_agg.h5
├── README.md
```

---

## 🔄 Workflow

1. Extract posture/keypoint information from video frames  
2. Perform feature extraction and normalization  
3. Generate sequential posture data  
4. Train LSTM-based deep learning model  
5. Perform real-time fall detection on video streams  

---

## 👩‍💻 Author

**Yeturu Thanvi**  
