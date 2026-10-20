FROM python:3.12-slim-bookworm

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Native dependencies for MediaPipe, OpenCV and XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgles2 \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-landmark.txt .

RUN pip install --no-cache-dir -r requirements-landmark.txt

# Files downloaded into models/ by Cloud Build.
COPY models/ ./models/

COPY landmark_api.py ./landmark_api.py

CMD ["sh", "-c", "exec python -m uvicorn landmark_api:app --host 0.0.0.0 --port \"$PORT\""]
