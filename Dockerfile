FROM python:3.10-slim

WORKDIR /app

# ✅ 1. Copy ONLY requirements first
COPY requirements-api.txt .
COPY requirements-train.txt .


RUN pip install --upgrade pip

RUN pip install -r requirements-api.txt
RUN pip install -r requirements-train.txt


# ✅ 2. Copy the rest of the project
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]