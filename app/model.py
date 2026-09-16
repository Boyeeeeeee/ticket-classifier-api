"""
Loads the fine-tuned classifier and exposes a simple predict() function.

TODO (after training/train.py has been run):
- point MODEL_DIR at the saved model directory (e.g. "training/output/final_model")
- fill in LABELS with the actual category names in the order the model was trained on
"""
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "training/output/final_model"  # TODO: update after training
LABELS: list[str] = []  # TODO: fill in with actual class names, in label-index order


class TicketClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.labels = LABELS or [
            f"label_{i}" for i in range(self.model.config.num_labels)
        ]

    @torch.no_grad()
    def predict(self, text: str) -> list[tuple[str, float]]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        ranked = sorted(zip(self.labels, probs), key=lambda x: x[1], reverse=True)
        return ranked


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    """Load the model once per process; FastAPI dependency will reuse this."""
    return TicketClassifier()
