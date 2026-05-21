"""
Uso:
    python capture_data.py                    # captura para ASL (por defecto)
    python capture_data.py --country colombia # captura para Colombia
    python capture_data.py --country mexico   # captura para México

Controles:
    A-Z        → selecciona la letra a capturar
    ESC        → salir
"""
import argparse
import csv
import os

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

SAMPLES_PER_KEY = 100
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'space']

COUNTRY_LABELS = {
    'asl':      'ASL (Estados Unidos)',
    'colombia': 'Colombia',
    'China':   'China',
}

fieldnames = [f'{a}{i}' for i in range(21) for a in ['x', 'y', 'z']] + ['label']


def run(country: str):
    if country not in COUNTRY_LABELS:
        raise ValueError(f"País '{country}' no válido. Opciones: {list(COUNTRY_LABELS.keys())}")

    output_dir = os.path.join('data', country)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'landmarks.csv')

    file_exists = os.path.isfile(output_csv)
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    current_label = None
    count = 0

    print(f"\n  País: {COUNTRY_LABELS[country]}")
    print(f"  Guardando en: {output_csv}")
    print("\n  Instrucciones:")
    print("    Presiona A-Z para capturar esa seña")
    print("    Presiona ESC para salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            if current_label is not None and count < SAMPLES_PER_KEY:
                row = {}
                for i, lm in enumerate(hand.landmark):
                    row[f'x{i}'] = lm.x
                    row[f'y{i}'] = lm.y
                    row[f'z{i}'] = lm.z
                row['label'] = current_label
                writer.writerow(row)
                csv_file.flush()
                count += 1

        country_label = COUNTRY_LABELS[country]
        cv2.putText(frame, f'Pais: {country_label}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        if current_label:
            status = f'Letra: {current_label} | {count}/{SAMPLES_PER_KEY}'
            color = (0, 255, 0) if result.multi_hand_landmarks else (0, 0, 255)
            cv2.putText(frame, status, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if count >= SAMPLES_PER_KEY:
                cv2.putText(frame, 'Listo! Presiona otra letra', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'Presiona una letra para empezar', (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        cv2.imshow(f'Captura: {country_label}', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key != 255:
            char = chr(key).upper()
            if char in LETTERS:
                current_label = char
                count = 0
                print(f"  Capturando: {char}")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Datos guardados en {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capturar landmarks de lenguaje de señas')
    parser.add_argument('--country', default='asl',
                        choices=list(COUNTRY_LABELS.keys()),
                        help='País / lengua de señas a capturar')
    args = parser.parse_args()
    run(args.country)