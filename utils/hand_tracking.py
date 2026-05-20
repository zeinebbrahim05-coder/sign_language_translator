import cv2
import mediapipe as mp

class HandDetector:

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        self.hand = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def run(self):
        tip_ids = [4, 8, 12, 16, 20]

        while True:

            success, frame = self.cap.read()

            if success:

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = self.hand.process(rgb_frame)

                if result.multi_hand_landmarks:
                    if result.multi_handedness:
                        label = result.multi_handedness[0].classification[0].label
                    else:
                        label = "Unknown"

                    for hand_landmarks in result.multi_hand_landmarks:

                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        for id, lm in enumerate(hand_landmarks.landmark):

                            h, w, c = frame.shape

                            cx, cy = int(lm.x * w), int(lm.y * h)

                            cv2.circle(
                                frame,
                                (cx, cy),
                                5,
                                (255, 0, 0),
                                cv2.FILLED
                            )
                        fingers = []

                        # Thumb
                        if label == "Right":
                            thumb = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
                        else:
                            thumb = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x

                        fingers.append(1 if thumb else 0)

                        # Other fingers
                        for tip in tip_ids[1:]:

                            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                                fingers.append(1)

                            else:
                                fingers.append(0)

                        print("Fingers:", fingers.count(1))
                        cv2.putText(
                            frame,
                            str(fingers.count(1)),
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3,
                            (0, 0, 255),
                            5
                        )
                        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                            print("Index Finger Open")

                cv2.imshow("Hand Tracking", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()