FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# The trained model is no longer baked into the image. app/model.py
# downloads it from the Hugging Face Hub at process startup
# (see MODEL_DIR in app/model.py), so the image stays small and the
# model can be updated by pushing a new version to the Hub without
# rebuilding this container.
#


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]