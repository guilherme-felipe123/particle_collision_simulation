# api/main.py
import json
import torch
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from ml.normalizer import Normalizer
from ml.deepset_model import DeepSetModel
from ml.model_manager import load_best_model

app = FastAPI()


normalizer = Normalizer()

try:
    normalizer.load("ml/normalization.json")
except FileNotFoundError:
    print("⚠️ normalization not found, model not ready")



# Load model once (important)
model = DeepSetModel()
model = load_best_model(model)
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
        x = normalizer.transform(x)

        prediction = model(x)
        prediction = prediction.item()

    return {
        "predicted_particles": prediction
    }