#!/bin/bash

set -e

IMAGE_NAME="particle-sim"

echo "🚀 Starting setup..."

# 1. Build Docker image if not exists
if [[ -z "$(docker images -q $IMAGE_NAME 2> /dev/null)" ]]; then
    echo "📦 Building Docker image..."
    docker build -t $IMAGE_NAME .
else
    echo "✅ Docker image already exists"
fi

# 2. Generate data if not exists
if [ ! -f "data/events.jsonl" ]; then
    echo "📊 Generating dataset..."
    docker run -it \
      --env PYTHONPATH=/app \
      --volume $(pwd):/app \
      $IMAGE_NAME \
      python simulation/generate_data.py
else
    echo "✅ Dataset already exists"
fi

# 3. Train model if not exists
if [ ! -f "ml/model.pth" ]; then
    echo "🧠 Training model..."
    docker run -it \
      --env PYTHONPATH=/app \
      --volume $(pwd):/app \
      $IMAGE_NAME \
      python ml/train.py
else
    echo "✅ Model already exists"
fi

# 4. Run API
if [ "$(docker ps -q -f name=particle-api)" ]; then
    echo "✅ API already running"
else
    docker rm -f particle-api 2>/dev/null || true
    echo "🌐 Starting API..."
    docker run -d \
      --name particle-api \
      --env PYTHONPATH=/app \
      --volume $(pwd):/app \
      -p 8000:8000 \
      particle-sim \
      uvicorn api.main:app --host 0.0.0.0 --port 8000
fi

# 5. Run simulation
echo "🎮 Starting simulation..."
xhost +local:docker

docker run -it \
  --network=host \
  --env DISPLAY=$DISPLAY \
  --env PYTHONPATH=/app \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume $(pwd):/app \
  $IMAGE_NAME \
  python simulation/main.py