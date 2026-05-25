# Sign Language Translator

> Real-time sign language translation to text and speech using computer vision and machine learning.

## 📌 Overview

This project bridges the communication gap between sign language users and non-signers by detecting hand gestures through a webcam and converting them into **spoken language** in real time.

The pipeline goes from raw camera input → hand landmark extraction via MediaPipe → gesture classification via a trained Random Forest model → text-to-speech output via pyttsx3.

## Current features:
- ✅ Real-time hand detection and tracking
- ✅ Finger counting (0–5 fingers)
- ✅ Left/right hand orientation detection
- ✅ Gesture recognition (RandomForest classifier)
- ✅ Text-to-speech output (voice)
- ✅ Confidence threshold (70% for reliable predictions)

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10 |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe Hands |
| Machine Learning | scikit-learn (Random Forest) |
| Speech Synthesis | pyttsx3 |
| Data Handling | pandas, numpy |

## 📁 Project Structure

```
sign_language_translator/
├── data/
│   └── gestures.csv          # Collected gesture dataset
├── models/
│   └── gesture_model.pkl     # Trained classifier
├── data_collection.py        # Collect gesture samples via webcam
├── train_model.py            # Train and save the model
├── predict.py                # Run real-time gesture recognition
├── main.py                   # Entry point (uses HandDetector class)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/zeinebbrahim05-coder/sign_language_translator.git
cd sign_language_translator
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1 — Collect gesture data

Run the data collection script and enter the name of the gesture you want to record (e.g. `hello`, `yes`, `no`).

```bash
python data_collection.py
```

- Press `C` to capture a sample
- Press `Q` to quit
- Repeat for each gesture you want to teach the model (aim for 100+ samples per gesture)

Samples are saved to `data/gestures.csv`.

### Step 2 — Train the model

```bash
python train_model.py
```

This trains a Random Forest classifier on your collected data and saves the model to `models/gesture_model.pkl`. Accuracy is printed at the end.

### Step 3 — Run real-time prediction

```bash
python predict.py
```

Point your hand at the webcam. Recognized gestures are displayed on screen and spoken aloud. Press `Q` to quit.

## ⚙️ How It Works

1. Each video frame is captured and flipped for a mirror effect
2. MediaPipe extracts 21 hand landmarks (x, y coordinates = 42 features)
3. The trained Random Forest classifier predicts the gesture
4. If confidence is above 70%, the gesture label is displayed and spoken
5. A new thread handles text-to-speech to keep the video feed smooth

## 📋 Requirements

See `requirements.txt`. Main dependencies:

```
opencv-contrib-python
mediapipe
scikit-learn
pyttsx3
pandas
numpy
```

## ⚠️ Notes

- Works best in good lighting with a plain background
- One hand detected at a time
- Tested on Windows with Python 3.10
- `pyttsx3` uses the system's built-in TTS engine (SAPI5 on Windows)
