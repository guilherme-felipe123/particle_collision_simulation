# 🧪 Particle Collision Simulation with Deep Learning

## 📌 Overview

This project implements an end-to-end machine learning system for real-time particle collision reconstruction, combining simulation, model inference, and API-based deployment in a distributed architecture.

The system mimics, in a simplified way, how experiments like those at CERN infer what truly happened during a collision using imperfect observations.


---

## 🎥 Demo

[https://github.com/user-attachments/assets/02a080d4-77a5-4e13-a69c-800f46ac2be1](https://github.com/user-attachments/assets/02a080d4-77a5-4e13-a69c-800f46ac2be1)

---

## 🧠 Idea

In real particle physics experiments:

* Collisions produce multiple particles
* Detectors capture noisy and incomplete signals
* Machine learning is used to reconstruct the underlying event

This project follows a similar pipeline:

1. Simulate particle collisions
2. Add detector-like noise
3. Perform **real-time inference via an API**
4. Estimate the true number of particles

---

## 🏗️ System Architecture

This project is structured as a **distributed ML system**:

```
Pygame Simulation  →  FastAPI Service  →  PyTorch Model
        │                    │
        └──── sends data ────┘
```

* The simulation generates events in real time
* Data is sent to an API endpoint
* The model performs inference and returns predictions
* Results are displayed live

---

## ⚙️ How it works

### 🔬 Simulation

* Particle collisions are generated programmatically
* Each event produces a variable number of particles
* A detector layer introduces noise and partial observations

---

### 🤖 Model

* A **DeepSet-based neural network** processes unordered particle data
* The model learns to:
  → Estimate the number of particles from noisy inputs

---

### 🌐 API (FastAPI)

The trained model is deployed as a REST API using FastAPI. The service loads the model at startup and exposes a /predict endpoint for real-time inference, enabling decoupled communication between the simulation client and the ML model.

* The trained model is served via a REST API
* Endpoint: `/predict`
* Receives particle features and returns predictions

Example request:

```json
{
  "particles": [[px, py, energy], ...]
}
```

---

### 🔁 Training

* Synthetic dataset generated from the simulation
* Model trained using PyTorch

---

## 🚢 Deployment

The application is containerized using Docker to ensure reproducibility and portability across environments.

- Encapsulates simulation, model, and API dependencies
- Enables consistent execution across systems
- Supports local development and potential cloud deployment

The API service can be deployed independently, enabling scalable ML inference.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* FastAPI
* Docker
* Pygame (visualization)

---

## 🚀 How to run

### 1. Build Docker image

```bash
docker build -t particle-sim .
```

---

### 2. Generate Some Data

```bash
docker run -it \
  --env PYTHONPATH=/app \
  --volume $(pwd):/app \
  particle-sim \
  python simulation/generate_data.py
```

---

### 3. Train the model

```bash
docker run -it \
  --env PYTHONPATH=/app \
  --volume $(pwd):/app \
  particle-sim \
  python ml/train.py
```

---

### 4. Run API

```bash
docker run -it \
  --env PYTHONPATH=/app \
  --volume $(pwd):/app \
  -p 8000:8000 \
  particle-sim \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

### 5. Run simulation

```bash
xhost +local:docker

docker run -it \
  --env DISPLAY=$DISPLAY \
  --env PYTHONPATH=/app \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume $(pwd):/app \
  particle-sim \
  python simulation/main.py
```

---

## 📊 Current Status

* ✅ Simulation pipeline implemented
* ✅ Deep learning model trained
* ✅ FastAPI inference service working
* ✅ Real-time integration (simulation ↔ API)
* ⚠️ Model predictions still need improvement

---

## 📈 Future Improvements

* Improve model accuracy and stability
* Add richer physics-inspired features
* Implement better detector simulation
* Measure and optimize **inference latency**
* Add batching / async inference
* Build a monitoring dashboard

---

## 🧠 Key Learnings

* Handling **noisy scientific data**
* Designing **permutation-invariant models (DeepSets)**
* Building **ML inference APIs with FastAPI**
* Integrating **real-time systems with machine learning**
* Debugging **client-server data contracts (422 errors, schema mismatches)**



