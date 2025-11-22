# IA (FastAPI) Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Optional build tools if some packages need it
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements-light.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8000

# Expect GOOGLE_API_KEY / GEMINI_API_KEY and optionally GEMINI_MODEL in env
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
