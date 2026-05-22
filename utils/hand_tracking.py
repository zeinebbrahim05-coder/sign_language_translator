import cv2
import mediapipe as mp

from utils.gesture_recognition import detect_gesture


class HandDetector:

    def __init__(self):

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        # Hand detector
        self.hand = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Fingertips ids
        self.tip_ids = [4, 8, 12, 16, 20]

    def get_fingers_state(self, hand_landmarks, label):

        fingers = []

        # Thumb
        if label == "Right":
            thumb_open = (
                hand_landmarks.landmark[4].x
                < hand_landmarks.landmark[3].x
            )
        else:
            thumb_open = (
                hand_landmarks.landmark[4].x
                > hand_landmarks.landmark[3].x
            )

        fingers.append(1 if thumb_open else 0)

        # Other fingers
        for tip in self.tip_ids[1:]:

            finger_open = (
                hand_landmarks.landmark[tip].y
                < hand_landmarks.landmark[tip - 2].y
            )

            fingers.append(1 if finger_open else 0)

        return fingers

    def draw_landmarks(self, frame, hand_landmarks):

        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        h, w, c = frame.shape

        for lm in hand_landmarks.landmark:

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 0, 0),
                cv2.FILLED
            )

    def run(self):

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.hand.process(rgb_frame)

            if result.multi_hand_landmarks:

                # Right / Left hand
                if result.multi_handedness:
                    label = result.multi_handedness[0].classification[0].label
                else:
                    label = "Unknown"

                for hand_landmarks in result.multi_hand_landmarks:

                    # Draw hand
                    self.draw_landmarks(frame, hand_landmarks)

                    # Detect fingers
                    fingers = self.get_fingers_state(
                        hand_landmarks,
                        label
                    )

                    # Detect gesture
                    gesture = detect_gesture(fingers)

                    # Display gesture
                    cv2.putText(
                        frame,
                        gesture,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0, 0, 255),
                        5
                    )

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()