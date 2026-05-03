import json
import torch
from torch.utils.data import DataLoader

from ml.collate import collate_fn
from ml.model import CollisionModel
from data.dataset import CollisionDataset
from ml.deepset_model import DeepSetModel



def train():
    dataset = CollisionDataset("data/events.jsonl")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    all_particles = []
    
    for x, _ in loader:
        x = x.float()
        for sample in x:
            all_particles.append(sample)
    
    # Now concatenate ALL particles
    all_particles = torch.cat(all_particles, dim=0)  # [total_particles, features]
    
    mean = all_particles.mean(dim=0)
    std = all_particles.std(dim=0)

    with open("ml/normalization.json", "w") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist()
        }, f)

    model = DeepSetModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(10):
        total_loss = 0

        for x, _ in loader:
            x = x.float()
            for sample in x:
                all_particles.append(sample)

        for x, y in loader:
            x = x.float()
            y = y.float()

            x = (x - mean) / (std + 1e-8)

            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "ml/model.pth")


if __name__ == "__main__":
    train()