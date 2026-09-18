# Ticket Classifier API

A deployed NLP service that classifies customer complaint/support-ticket text
into product/issue categories, using a fine-tuned DistilBERT model served via
FastAPI, containerized with Docker, and tested with CI.

This project is deliberately engineering-forward: the model is a solid but
simple baseline, and most of the effort goes into making it a real, servable
piece of software — not another notebook.

## Problem

Given free-text customer complaint narratives, predict which product/issue
category the complaint belongs to. This mirrors real support-ticket auto-routing
systems used in customer service operations.

## Dataset

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
— public, real-world consumer complaint narratives labeled by product category.

## Model

Fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) on 8
merged product categories (near-duplicate CFPB taxonomy labels from different
years were merged before training — see `training/train.py`). The trained
model is hosted on the Hugging Face Hub:

**[Deboyeeeeeee/ticket-classifier-distilbert](https://huggingface.co/Deboyeeeeeee/ticket-classifier-distilbert)**

The API downloads it from the Hub at startup rather than bundling it in the
repo or the Docker image, keeping both lightweight.

## Architecture

```
Client -> FastAPI (/predict) -> DistilBERT (fine-tuned, loaded from HF Hub) -> category + confidence
```

- `training/` — data prep + fine-tuning script (run locally/Colab, not in this container)
- `app/` — FastAPI service that loads the trained model and serves predictions
- `tests/` — unit tests for preprocessing and API behavior
- `.github/workflows/ci.yml` — runs tests on every push
- `Dockerfile` — reproducible container for the API

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then POST to `/predict`:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My credit card company charged me twice for the same transaction"}'
```

## Deployment

TODO: deployed at <live URL> (Render/Fly.io/HF Spaces)

## Results

Evaluated on a held-out test set (15% split, 3,600 examples across 8 balanced
categories):

- **Test accuracy: 83.25%**
- **Test macro-F1: 0.8322**

Per-class F1 ranged from 0.77 (Credit reporting, credit repair services, or
other personal consumer reports) up to 0.93 (Student loan). The
lowest-performing classes were also the ones with the broadest, most varied
complaint narratives after label merging (Credit reporting, Debt collection),
while narrower categories (Student loan, Mortgage) were easiest to separate.

## What this project demonstrates

- Fine-tuning a transformer on real-world unstructured text data
- Wrapping a model in a production-shaped API (validation, error handling, schemas)
- Hosting and loading a trained model from a model registry (Hugging Face Hub)
- Containerization and reproducible deployment
- Automated testing and CI
- Basic request logging/monitoring on a live endpoint