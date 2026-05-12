# sign-language-detector

Detector de lenguaje de señas en tiempo real con Flask + MediaPipe + scikit-learn.

## Requisitos

- **Python 3.12** (obligatorio — los wheels de MediaPipe 0.10.14 están compilados para `cp312`. En 3.13 o 3.14 fallará la instalación).

## Instalación

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/macOS:
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```powershell
python app.py
```

Abre `http://127.0.0.1:5000` en el navegador.

## Entrenar el modelo

```powershell
python model/train.py
```

Lee `data/archive/asl_landmarks_final.csv` y guarda `classifier.pkl` y `label_encoder.pkl` en `model/`.

## Notas importantes

- **No instales `mediapipe` sin pin de versión** (`pip install mediapipe` a secas). La versión 0.10.35+ eliminó la API legacy `mp.solutions` en Windows, que es la que usa este proyecto. Usa siempre `pip install -r requirements.txt`.
- Si tras un `pip install` aparece `AttributeError: module 'mediapipe' has no attribute 'solutions'`, fuerza la versión correcta:
  ```powershell
  pip install --force-reinstall --no-cache-dir mediapipe==0.10.14
  ```
