from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartDiseaseInput

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational ML inference API. Not for clinical diagnosis.",
    version="1.0.0",
)

model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/info")
def info():
    return {
        "model_type": "StandardScaler + LogisticRegression pipeline",
        "target": "target (0=absence, 1=presence)",
        "features": FEATURES,
        "feature_count": len(FEATURES),
    }

@app.post("/predict")
def predict(payload: HeartDiseaseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_df = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])

    return {
        "heart_disease": bool(prediction),
        "prediction": prediction,
        "probability": probability,
        "message": (
            "The model predicted presence of heart disease."
            if prediction == 1
            else "The model predicted absence of heart disease."
        ),
        "disclaimer": "Educational demonstration only; not a medical diagnosis."
    }
