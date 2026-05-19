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

        while True:

            success, frame = self.cap.read()

            if success:

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = self.hand.process(rgb_frame)

                if result.multi_hand_landmarks:

                    for hand_landmarks in result.multi_hand_landmarks:

                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        for id, lm in enumerate(hand_landmarks.landmark):

                            h, w, c = frame.shape

                            cx, cy = int(lm.x * w), int(lm.y * h)

                            print(id, cx, cy)

                            cv2.circle(
                                frame,
                                (cx, cy),
                                5,
                                (255, 0, 0),
                                cv2.FILLED
                            )

                cv2.imshow("Hand Tracking", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()