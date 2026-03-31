# 🧪 Particle Collision Simulation with Deep Learning

## 📌 Overview
This project simulates particle collisions and applies a deep learning model to reconstruct information from noisy detector data — inspired by real-world high-energy physics experiments.

The goal is to mimic, in a simplified way, how experiments like those at CERN infer what truly happened during a collision using imperfect observations.

---

## 🎥 Demo

https://github.com/user-attachments/assets/02a080d4-77a5-4e13-a69c-800f46ac2be1

---

## 🧠 Idea

In real particle physics experiments:
- Collisions produce multiple particles
- Detectors capture noisy and incomplete signals
- Machine learning is used to reconstruct the underlying event

This project follows a similar pipeline:

1. Simulate particle collisions
2. Add detector-like noise
3. Use a neural network to estimate the true number of particles

---

## ⚙️ How it works

### 🔬 Simulation
- Particle collisions are generated programmatically
- Each event produces a variable number of particles
- A detector layer introduces noise and partial observations

### 🤖 Model
- A **DeepSet-based neural network** processes unordered particle data
- The model learns to:
  → Estimate the number of particles from noisy inputs

### 🔁 Training
- Synthetic dataset generated from the simulation
- Model trained using PyTorch

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Docker
- Pygame (visualization)

---

## 🚀 How to run


### 1. Build Docker image

```bash
docker build -t particle-sim .
```

### 2. Train the model
```bash
docker run -it \
  --env PYTHONPATH=/app \
  --volume $(pwd):/app \
  particle-sim \
  python ml/train.py
```

### 3. Run simulation
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
- ✅ Simulation pipeline implemented
- ✅ Deep learning model trained
- ⚠️ Model predictions still need improvement

---

🎯 Future Improvements
- Improve model accuracy and stability
- Add richer physics-inspired features
- Implement better detector simulation
- Explore real-time inference constraints


