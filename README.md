# Sign Language Translator

> Real-time sign language translation to text and speech using computer vision and machine learning.

## 📌 Overview

This project aims to bridge the communication gap between sign language users and non-signers by translating hand gestures into **spoken language** in real time.

**Current features:**
- ✅ Real-time hand detection and landmark tracking
- ✅ Finger counting (0–5 fingers)
- ✅ Left/right hand orientation detection
- ✅ Modular `HandDetector` class

**In development:**
- 🔄 Gesture dataset collection
- ⏳ Machine learning model training
- ⏳ Text-to-speech output
- ⏳ Real-time translation pipeline

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10 |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe Hands |
| Speech Synthesis | pyttsx3 (planned) |

## 📁 Project Structure
sign_language_translator/
├── utils/
│ └── hand_tracking.py
├── data/
├── .gitignore
├── main.py
└── README.md

text

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/zeinebbrahim05-coder/sign_language_translator.git
cd sign_language_translator

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe numpy

# Run the hand tracker
python main.py
Press q to exit.
