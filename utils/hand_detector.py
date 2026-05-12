import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def extract_landmarks(frame):
    result = hands.process(frame)
    if not result.multi_hand_landmarks:
        return None, None

    hand = result.multi_hand_landmarks[0]
    landmarks = [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]
    return np.array(landmarks), hand
