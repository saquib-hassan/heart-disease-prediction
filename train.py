from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET = "target"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Download heart.csv from the assignment's Kaggle "
            "dataset and place it at data/heart.csv."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Expected columns: {required}. "
            "Check that you downloaded the UCI-style heart.csv with target column."
        )

    df = df[required].dropna()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    unique_targets = set(y.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found {unique_targets}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
