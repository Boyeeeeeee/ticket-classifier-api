import logging
import time

from fastapi import Depends, FastAPI, HTTPException

from app.model import TicketClassifier, get_classifier
from app.schemas import HealthResponse, PredictionItem, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ticket-classifier")

app = FastAPI(
    title="Ticket Classifier API",
    description="Classifies customer complaint text into product/issue categories.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_classifier()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    classifier: TicketClassifier = Depends(get_classifier),
):
    start = time.time()
    try:
        ranked = classifier.predict(request.text)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "predict request | chars=%d | top_label=%s | latency_ms=%.1f",
        len(request.text),
        ranked[0][0],
        elapsed_ms,
    )

    predictions = [PredictionItem(label=lbl, confidence=round(p, 4)) for lbl, p in ranked]
    return PredictResponse(top_prediction=predictions[0], all_predictions=predictions)
