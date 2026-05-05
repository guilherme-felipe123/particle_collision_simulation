FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps only if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (better caching)
COPY requirements-api.txt requirements-train.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir -r requirements-train.txt

# Copy only necessary code
COPY api ./api
COPY ml ./ml
COPY data ./data
COPY simulation ./simulation
COPY detector ./detector

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]