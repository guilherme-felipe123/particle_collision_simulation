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

## 🔄 CI/CD

A GitHub Actions pipeline is used to:

- Validate code on each push
- Install dependencies automatically
- Build the Docker image for deployment

This ensures reproducibility and early detection of integration issues.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* FastAPI
* Docker
* Pygame (visualization)

---

## 🚀 How to run

### 🔹 Automatic Setup

Run the provided script:

```bash
chmod +x run_app.sh
./run_app.sh
```

This script will:

- Build the Docker image (if not already built)
    
- Generate the dataset (if missing)
    
- Train the model (if missing)
    
- Start the FastAPI service
    
- Launch the particle simulation


⚠️ The first run may take a few minutes due to data generation and model training.

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



