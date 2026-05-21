"""
Uso:
    python model/train.py                    # entrena con datos ASL base
    python model/train.py --country colombia # entrena modelo Colombia
    python model/train.py --country asl      # entrena modelo ASL
    python model/train.py --country mexico   # entrena modelo México
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

COUNTRY_DATA = {
    'asl': {
        'sources': [
            BASE_DIR / 'data' / 'archive' / 'asl_landmarks_final.csv',
            BASE_DIR / 'data' / 'asl' / 'landmarks.csv',
        ],
        'label': 'ASL (Estados Unidos)',
    },
    'colombia': {
        'sources': [
            BASE_DIR / 'data' / 'colombia' / 'landmarks.csv',
        ],
        'label': 'Colombia',
    },
    'china': {
        'sources': [
            BASE_DIR / 'data' / 'china' / 'landmarks.csv',
        ],
        'label': 'China',
    },
}


def train(country: str):
    if country not in COUNTRY_DATA:
        raise ValueError(f"País '{country}' no registrado. Opciones: {list(COUNTRY_DATA.keys())}")

    config = COUNTRY_DATA[country]
    print(f"\n=== Entrenando modelo: {config['label']} ===\n")

    dfs = []
    for path in config['sources']:
        if path.exists():
            df_part = pd.read_csv(path)
            print(f"  Cargado: {path.name} → {len(df_part)} filas")
            dfs.append(df_part)
        else:
            print(f"  Saltando (no existe): {path}")

    if not dfs:
        raise FileNotFoundError(
            f"No se encontraron datos para '{country}'.\n"
            f"Captura datos con: python capture_data.py --country {country}"
        )

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total filas: {len(df)}")

    X = df.drop(columns=['label']).values
    y = df['label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    out_dir = BASE_DIR / 'model' / country
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(out_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"  Modelo guardado en: {out_dir}/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo de lenguaje de señas')
    parser.add_argument('--country', default='asl',
                        choices=list(COUNTRY_DATA.keys()),
                        help='País / lengua de señas a entrenar')
    args = parser.parse_args()
    train(args.country)