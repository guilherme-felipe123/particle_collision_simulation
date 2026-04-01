# api/main.py

import torch
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from ml.deepset_model import DeepSetModel

app = FastAPI()

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
        prediction = model(x).item()

    return {
        "predicted_particles": prediction
    }