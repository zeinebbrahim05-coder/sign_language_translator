import streamlit as st
import cv2
import mediapipe as mp
import pickle
import pyttsx3
import threading
import time

# =========================
# TEXT TO SPEECH
# =========================

is_speaking = False

def speak_text(text):

    global is_speaking

    is_speaking = True

    engine = pyttsx3.init()

    engine.setProperty('rate', 150)

    engine.say(text)

    engine.runAndWait()

    engine.stop()

    is_speaking = False

# =========================
# LOAD MODEL
# =========================

with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hand = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# STREAMLIT UI
# =========================

st.title("Sign Language Translator AI")

frame_placeholder = st.empty()

# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)

last_spoken = None

while True:

    success, frame = camera.read()

    if not success:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hand.process(rgb_frame)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # EXTRACT LANDMARKS
            # =========================

            data = []

            for lm in hand_landmarks.landmark:

                data.append(lm.x)
                data.append(lm.y)

            # =========================
            # PREDICTION
            # =========================

            prediction = model.predict([data])[0]

            probabilities = model.predict_proba([data])[0]

            confidence = max(probabilities)

            # =========================
            # UNKNOWN DETECTION
            # =========================

            if confidence < 0.7:

                text = "UNKNOWN"

                color = (0, 0, 255)

            else:

                text = f"{prediction} ({confidence*100:.1f}%)"

                color = (0, 255, 0)

                # =========================
                # VOICE OUTPUT
                # =========================

                if prediction != last_spoken and not is_speaking:

                    last_spoken = prediction

                    threading.Thread(
                        target=speak_text,
                        args=(prediction,),
                        daemon=True
                    ).start()

            # =========================
            # DISPLAY TEXT
            # =========================

            cv2.putText(
                frame,
                text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

    else:
        last_spoken = None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame, channels="RGB")

camera.release()