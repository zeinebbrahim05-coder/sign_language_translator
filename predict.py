import cv2
import mediapipe as mp
import pickle
import pyttsx3
import time
import threading

# =========================
# TEXT TO SPEECH SETUP
# =========================

is_speaking = False

def speak_text(text):
    global is_speaking
    is_speaking = True
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    is_speaking = False

# =========================
# LOAD TRAINED MODEL
# =========================

with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# MEDIAPIPE SETUP
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
# CAMERA SETUP
# =========================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# VARIABLES
# =========================

last_spoken_gesture = None

# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(rgb_frame)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

            raw_prediction = model.predict([data])[0]
            probabilities = model.predict_proba([data])[0]
            confidence = max(probabilities)

            if confidence < 0.7:
                display_text = "UNKNOWN"
                color = (0, 0, 255)
                should_speak = False
            else:
                display_text = f"{raw_prediction} ({confidence * 100:.1f}%)"
                color = (0, 255, 0)
                should_speak = True

            # =========================
            # VOICE OUTPUT
            # =========================

            if should_speak and raw_prediction != last_spoken_gesture and not is_speaking:
                last_spoken_gesture = raw_prediction
                threading.Thread(
                    target=speak_text,
                    args=(raw_prediction,),
                    daemon=True
                ).start()

            cv2.putText(frame, display_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    else:
        last_spoken_gesture = None

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()