import os

os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import pickle
import socket
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template

from utils.hand_detector import extract_landmarks, mp_drawing, mp_hands

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'classifier.pkl'
ENCODER_PATH = BASE_DIR / 'model' / 'label_encoder.pkl'

if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    raise FileNotFoundError(
        "No se encontraron los archivos del modelo. "
        "Ejecuta primero: python model/train.py"
    )

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

app = Flask(__name__)


def _placeholder_frame(message: str) -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + _placeholder_frame('No se detecto camara') + b'\r\n')
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, hand_landmarks = extract_landmarks(frame_rgb)

            if landmarks is not None:
                prediction = model.predict([landmarks])[0]
                label = le.inverse_transform([prediction])[0]

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 4, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'


if __name__ == '__main__':
    port = 5000
    print(f"\n  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{_lan_ip()}:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
