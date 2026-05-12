import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'archive' / 'asl_landmarks_final.csv'
MODEL_DIR = BASE_DIR / 'model'

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['label']).values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open(MODEL_DIR / 'classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"Modelo guardado en {MODEL_DIR}/")
