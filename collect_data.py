import cv2
import csv
import mediapipe as mp
import time

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- SETTINGS ----------------
current_label = None
sample_count = 0
MAX_SAMPLES = 200
DELAY = 0.08

last_capture_time = 0

# ---------------- LABEL MAP ----------------
"""
A-Z COLLECTION

Press:
a -> A
b -> B
c -> C
...
z -> Z
"""

# ---------------- CSV ----------------
with open("data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        data = None

        # ---------------- HAND DETECTION ----------------
        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                # NORMALIZATION
                base_x = hand.landmark[0].x
                base_y = hand.landmark[0].y

                data = []

                for lm in hand.landmark:
                    data.append(lm.x - base_x)
                    data.append(lm.y - base_y)

                # DRAW LANDMARKS
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

        # ---------------- UI ----------------
        cv2.putText(
            frame,
            f"Label: {current_label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{MAX_SAMPLES}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press A-Z keys | Q = Quit",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Alphabet Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # ---------------- SET LABEL ----------------
        if ord('a') <= key <= ord('z'):

            current_label = chr(key).upper()
            sample_count = 0

            print(f"\nCollecting for alphabet: {current_label}")

        # ---------------- AUTO SAVE ----------------
        if data is not None and current_label is not None:

            current_time = time.time()

            if (
                current_time - last_capture_time > DELAY
                and sample_count < MAX_SAMPLES
            ):

                writer.writerow(data + [current_label])

                sample_count += 1
                last_capture_time = current_time

                print(
                    f"Saved {sample_count}/{MAX_SAMPLES} "
                    f"for {current_label}"
                )

        # ---------------- DONE ----------------
        if sample_count == MAX_SAMPLES:

            cv2.putText(
                frame,
                "DONE! Change Alphabet",
                (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            cv2.imshow("Alphabet Data Collection", frame)

        # ---------------- EXIT ----------------
        if key == ord('q'):
            break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()