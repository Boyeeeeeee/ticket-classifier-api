FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
# The trained model is copied in at build time once training is done.
# For a smaller image, consider downloading the model at container startup
# from a model registry (e.g. Hugging Face Hub) instead of baking it in.
COPY training/output/final_model/ ./training/output/final_model/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
