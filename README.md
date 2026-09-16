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

## Architecture

```
Client -> FastAPI (/predict) -> DistilBERT (fine-tuned) -> category + confidence
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

TODO: fill in after training — accuracy/F1 on held-out test set, per-class breakdown,
confusion matrix, error analysis.

## What this project demonstrates

- Fine-tuning a transformer on real-world unstructured text data
- Wrapping a model in a production-shaped API (validation, error handling, schemas)
- Containerization and reproducible deployment
- Automated testing and CI
- Basic request logging/monitoring on a live endpoint
