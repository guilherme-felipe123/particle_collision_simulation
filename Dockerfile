FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Copy requirements separately for better caching
COPY requirements-api.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-api.txt

COPY requirements-train.txt .
RUN pip install -r requirements-train.txt

# Copy project
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]