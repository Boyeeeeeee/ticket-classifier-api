from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw complaint / support ticket text to classify.",
        examples=["My credit card company charged me twice for the same transaction"],
    )


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    top_prediction: PredictionItem
    all_predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
