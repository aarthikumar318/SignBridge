import cv2
import numpy as np
import pickle
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify
import mediapipe as mp

app = Flask(__name__)

# ---------------- GLOBAL ----------------
latest_prediction = "Waiting..."

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- SMOOTHING ----------------
pred_buffer = deque(maxlen=5)

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/translator')
def translator():
    return render_template('translator.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')
@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/get_prediction')
def get_prediction():
    global latest_prediction
    return jsonify({"text": latest_prediction})

# ---------------- VIDEO STREAM ----------------
@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_prediction

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                latest_prediction = "NO HAND"
            else:
                hand = results.multi_hand_landmarks[0]

                base_x = hand.landmark[0].x
                base_y = hand.landmark[0].y

                data = []
                for lm in hand.landmark:
                    data.append(lm.x - base_x)
                    data.append(lm.y - base_y)

                data = np.array(data).reshape(1, -1)
                data = scaler.transform(data)

                pred = model.predict(data)[0]
                proba = model.predict_proba(data)[0]
                confidence = max(proba)

                if confidence < 0.5:
                    latest_prediction = "ADJUST HAND"
                else:
                    pred_buffer.append(pred)
                    final_pred = Counter(pred_buffer).most_common(1)[0][0]
                    latest_prediction = str(final_pred)

                # Draw on video
                cv2.putText(frame, latest_prediction, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            ret, buffer = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)