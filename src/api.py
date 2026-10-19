
from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_model.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run python src\\train.py first."
    )

model = joblib.load(MODEL_PATH)

# Keep feature names aligned with the pipeline used in training.
FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]


class HeartFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=800)
    thalach: float = Field(gt=0, le=300)
    oldpeak: float = Field(ge=0, le=15)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    exang: int = Field(ge=0, le=1)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational ML inference API; not for clinical decisions.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: HeartFeatures):
    try:
        row = pd.DataFrame(
            [[getattr(features, name) for name in FEATURES]],
            columns=FEATURES,
        )
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        logger.info("Prediction completed")
        return {
            "prediction": prediction,
            "probability_disease": round(probability, 4),
            "model_file": MODEL_PATH.name,
            "disclaimer": "Educational use only; not a medical diagnosis.",
        }
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500, detail="Prediction failed"
        )
