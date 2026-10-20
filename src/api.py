from pathlib import Path
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

REQUESTS = Counter(
    "heart_api_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)
ERRORS = Counter(
    "heart_api_errors_total",
    "HTTP requests resulting in client or server errors",
    ["endpoint", "status"],
)
PREDICTIONS = Counter(
    "heart_api_predictions_total",
    "Successful predictions by predicted class",
    ["prediction"],
)


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


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    route = "unmatched"

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        logger.exception("Unhandled HTTP request error")
        raise
    finally:
        matched_route = request.scope.get("route")
        if matched_route is not None:
            route = getattr(matched_route, "path", route)

        elapsed = time.perf_counter() - start
        method = request.method

        REQUESTS.labels(method, route, str(status)).inc()
        LATENCY.labels(method, route).observe(elapsed)

        if status >= 400:
            ERRORS.labels(route, str(status)).inc()

        logger.info(
            "method=%s endpoint=%s status=%s latency_seconds=%.4f",
            method, route, status, elapsed,
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict")
def predict(features: HeartFeatures):
    try:
        row = pd.DataFrame(
            [[getattr(features, name) for name in FEATURES]],
            columns=FEATURES,
        )
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])

        PREDICTIONS.labels(str(prediction)).inc()
        logger.info("Prediction completed; prediction=%s", prediction)

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
