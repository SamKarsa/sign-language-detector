import os

os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import pickle
import socket
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from utils.hand_detector import extract_landmarks, mp_drawing, mp_hands

BASE_DIR = Path(__file__).resolve().parent
MODEL_BASE = BASE_DIR / 'model'

COUNTRIES = {
    'asl':      {'name': 'ASL (Estados Unidos)', 'flag': '🇺🇸'},
    'colombia': {'name': 'Colombia',              'flag': '🇨🇴'},
    'china':   {'name': 'China',                'flag': 'Ch'},
}

_model_lock = threading.Lock()
current_country = 'asl'
model = None
le = None


def load_model(country: str):
    global model, le, current_country
    model_dir = MODEL_BASE / country
    classifier_path = model_dir / 'classifier.pkl'
    encoder_path    = model_dir / 'label_encoder.pkl'

    if not classifier_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"No se encontró modelo para '{country}'. "
            f"Ejecuta: python model/train.py --country {country}"
        )

    with open(classifier_path, 'rb') as f:
        new_model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        new_le = pickle.load(f)

    with _model_lock:
        model = new_model
        le = new_le
        current_country = country


def get_available_countries():
    available = {}
    for key, meta in COUNTRIES.items():
        model_dir = MODEL_BASE / key
        has_model = (model_dir / 'classifier.pkl').exists() and \
                    (model_dir / 'label_encoder.pkl').exists()
        available[key] = {**meta, 'available': has_model}
    return available


load_model('asl')

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
                with _model_lock:
                    _model = model
                    _le = le

                prediction = _model.predict([landmarks])[0]
                label = _le.inverse_transform([prediction])[0]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(frame, label, (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255, 0, 255), 4, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()


@app.route('/')
def index():
    countries = get_available_countries()
    return render_template('index.html',
                           countries=countries,
                           current_country=current_country)


@app.route('/set_country', methods=['POST'])
def set_country():
    data = request.get_json()
    country = data.get('country', '').strip()

    if country not in COUNTRIES:
        return jsonify({'ok': False, 'error': 'País no válido'}), 400

    try:
        load_model(country)
        return jsonify({
            'ok': True,
            'country': country,
            'name': COUNTRIES[country]['name'],
            'flag': COUNTRIES[country]['flag'],
        })
    except FileNotFoundError as e:
        return jsonify({'ok': False, 'error': str(e)}), 404


@app.route('/current_country')
def get_current_country():
    return jsonify({
        'country': current_country,
        **COUNTRIES[current_country]
    })


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