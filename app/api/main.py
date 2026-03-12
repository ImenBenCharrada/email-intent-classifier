from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.predict import IntentPredictor


app = FastAPI(title="Email Intent Classifier API", version="1.0.0")

predictor = IntentPredictor()


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Email text to classify")


class PredictionResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    result = predictor.predict(request.text)
    return PredictionResponse(**result)

    