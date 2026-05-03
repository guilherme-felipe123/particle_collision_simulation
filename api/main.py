# api/main.py
import json
import torch
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from ml.deepset_model import DeepSetModel

app = FastAPI()

with open("ml/normalization.json") as f:
    stats = json.load(f)

mean = torch.tensor(stats["mean"])
std = torch.tensor(stats["std"])

mean = mean.float()
std = std.float()

# Load model once (important)
model = DeepSetModel()
model.load_state_dict(torch.load("ml/model.pth", map_location="cpu"))
model.eval()

class ParticlesInput(BaseModel):
    particles: List[List[float]]

@app.get("/")
def root():
    return {"message": "Particle Collision API is running"}



@app.post("/predict")
def predict(data: ParticlesInput):
    x = torch.tensor([data.particles], dtype=torch.float32)

    with torch.no_grad():
        x = (x - mean) / (std + 1e-8)

        prediction = model(x)
        prediction = prediction.item()

    return {
        "predicted_particles": prediction
    }