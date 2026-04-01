from pydantic import BaseModel

class ParticlesInput(BaseModel):
    particles: list

@app.post("/predict")
def predict(data: ParticlesInput):
    x = torch.tensor([data.particles], dtype=torch.float32)