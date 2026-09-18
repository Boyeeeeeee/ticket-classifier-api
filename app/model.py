
import json
from functools import lru_cache

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# TODO: replace with your actual HF Hub repo id, e.g. "yourname/ticket-classifier-distilbert"
MODEL_DIR = "Deboyeeeeeee/ticket-classifier-distilbert"


class TicketClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # labels.json isn't a standard HF file, so fetch it explicitly.
        labels_path = hf_hub_download(repo_id=model_dir, filename="labels.json")
        with open(labels_path) as f:
            self.labels = json.load(f)

    @torch.no_grad()
    def predict(self, text: str) -> list[tuple[str, float]]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        # DistilBERT has no segment embeddings, so its forward() doesn't
        # accept token_type_ids even though some tokenizer versions include
        # it by default. Drop it if present rather than erroring.
        inputs.pop("token_type_ids", None)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        ranked = sorted(zip(self.labels, probs), key=lambda x: x[1], reverse=True)
        return ranked


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    """Load the model once per process; FastAPI dependency will reuse this."""
    return TicketClassifier()