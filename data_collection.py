import cv2
import mediapipe as mp
import csv
import os

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Hand detector
hand = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Gesture name
gesture_name = input("Enter gesture name: ")

# CSV path
csv_path = "data/gestures.csv"

# Check if file exists
file_exists = os.path.isfile(csv_path)

# Open CSV file
csv_file = open(csv_path, mode="a", newline="")

writer = csv.writer(csv_file)

# Create header if file is new
if not file_exists:

    header = ["gesture"]

    for i in range(21):

        header.append(f"x{i}")
        header.append(f"y{i}")

    writer.writerow(header)

    print("CSV file created")

print(f"Collecting data for: {gesture_name}")
print("Press 'c' to capture")
print("Press 'q' to quit")

# Variables
sample_count = 0
capture_flag = False
cooldown_counter = 0

# Main loop
while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    result = hand.process(rgb_frame)

    # Cooldown
    if cooldown_counter > 0:

        cooldown_counter -= 1

        capture_flag = False

    # If hand detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw points
            h, w, _ = frame.shape

            for lm in hand_landmarks.landmark:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    3,
                    (0, 255, 0),
                    cv2.FILLED
                )

            # Capture data
            if capture_flag and cooldown_counter == 0:

                row = [gesture_name]

                for lm in hand_landmarks.landmark:

                    row.append(lm.x)
                    row.append(lm.y)

                writer.writerow(row)

                sample_count += 1

                print(f"Captured sample #{sample_count}")

                capture_flag = False

                cooldown_counter = 10

    # Display sample count
    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show window
    cv2.imshow("Data Collection", frame)

    # Keyboard
    key = cv2.waitKey(1)

    # Capture
    if key == ord('c'):

        capture_flag = True

    # Quit
    elif key == ord('q'):

        break

# Cleanup
csv_file.close()

cap.release()

cv2.destroyAllWindows()

print("\nCollection complete")
print(f"Gesture: {gesture_name}")
print(f"Total samples: {sample_count}")
print(f"Saved to: {csv_path}")