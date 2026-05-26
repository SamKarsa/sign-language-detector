import os
import sys
import base64

os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import pickle
import threading
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import COUNTRIES, MODEL_BASE
from utils.hand_detector import extract_landmarks

# ── Estado del modelo ────────────────────────────────────────────────────────
_model_lock = threading.Lock()
current_country = 'asl'
model = None
le = None


def load_model(country: str):
    global model, le, current_country
    classifier_path = MODEL_BASE / country / 'classifier.pkl'
    encoder_path    = MODEL_BASE / country / 'label_encoder.pkl'

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


# ── Predicción desde frame enviado por el browser ───────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame', '')
    letter = data.get('letter', '').upper()

    try:
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'detected': False})

    if frame is None:
        return jsonify({'detected': False})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lm, hand_lm = extract_landmarks(rgb)

    if lm is None:
        return jsonify({'detected': False, 'label': None, 'score': 0.0, 'landmarks': None})

    with _model_lock:
        m, enc = model, le

    proba = m.predict_proba([lm])[0]
    label = enc.classes_[proba.argmax()]

    # Landmarks normalizados [0,1] para dibujar en el browser
    landmarks = [{'x': float(pt.x), 'y': float(pt.y)} for pt in hand_lm.landmark]

    score_val = float(proba.max())
    if letter and letter in enc.classes_:
        target_idx = list(enc.classes_).index(letter)
        score_val = float(proba[target_idx])

    return jsonify({
        'detected': True,
        'label': label,
        'score': score_val,
        'landmarks': landmarks,
    })


# ── Rutas principales ────────────────────────────────────────────────────────
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
    return jsonify({'country': current_country, **COUNTRIES[current_country]})


# ── Rutas de aprendizaje ─────────────────────────────────────────────────────
@app.route('/learn')
def learn():
    countries = get_available_countries()
    with _model_lock:
        letters = sorted([c for c in le.classes_ if len(c) == 1 and c.isalpha()])
    return render_template('learn.html',
                           countries=countries,
                           current_country=current_country,
                           letters=letters)


# ── Pista visual (landmarks de referencia desde los datos de entrenamiento) ──
@app.route('/learn/hint/<letter>')
def learn_hint(letter):
    import pandas as pd
    from config import DATA_SOURCES
    letter = letter.upper()

    with _model_lock:
        country = current_country

    dfs = []
    for path in DATA_SOURCES[country]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                subset = df[df['label'] == letter]
                if not subset.empty:
                    dfs.append(subset)
            except Exception:
                pass

    if not dfs:
        return jsonify({'ok': False, 'error': f'Sin datos para {letter}'})

    df = pd.concat(dfs, ignore_index=True)
    points = [{'x': float(df[f'x{i}'].mean()), 'y': float(df[f'y{i}'].mean())}
              for i in range(21)]

    return jsonify({'ok': True, 'letter': letter, 'points': points})


# ── Dashboard de progreso ────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    countries = get_available_countries()
    return render_template('dashboard.html',
                           countries=countries,
                           current_country=current_country)


# ── Modo deletrear palabras ──────────────────────────────────────────────────
SPELL_WORDS = {
    'asl':      ['LOVE', 'HELP', 'WATER', 'HAND', 'SIGN', 'LEARN',
                 'FOOD', 'HOME', 'GOOD', 'BOOK', 'PLAY', 'WORK'],
    'colombia': ['HOLA', 'AMOR', 'AGUA', 'MANO', 'LUNA', 'CASA',
                 'FLOR', 'BIEN', 'SOL', 'MAR', 'PAZ', 'LUZ'],
    'china':    ['LOVE', 'HELP', 'WATER', 'HAND', 'SIGN', 'LEARN',
                 'FOOD', 'HOME', 'GOOD', 'BOOK', 'PLAY', 'WORK'],
}

@app.route('/spell')
def spell():
    countries = get_available_countries()
    with _model_lock:
        country = current_country
        available = set(enc_class for enc_class in le.classes_)

    words = [w for w in SPELL_WORDS.get(country, SPELL_WORDS['asl'])
             if all(c in available for c in w)]

    return render_template('spell.html',
                           countries=countries,
                           current_country=country,
                           words=words)


# ── Arranque ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import socket
    def _lan_ip():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'

    port = int(os.environ.get('PORT', 5000))
    print(f"\n  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{_lan_ip()}:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
