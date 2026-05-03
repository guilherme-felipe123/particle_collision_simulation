import os
import json
import torch
from torch.utils.data import DataLoader

from ml.collate import collate_fn
from ml.normalizer import Normalizer
from data.dataset import CollisionDataset
from ml.deepset_model import DeepSetModel


def get_next_version():
    path = "ml/models/metadata.json"

    if not os.path.exists(path):
        return 1

    with open(path) as f:
        data = json.load(f)

    return data["latest_version"] + 1

def train():
    dataset = CollisionDataset("data/events.jsonl")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    # ✅ Fit normalizer ONCE
    normalizer = Normalizer()
    normalizer.fit(loader)
    normalizer.save("ml/normalization.json")

    model = DeepSetModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(10):
        total_loss = 0

        for x, y in loader:
            x = x.float()
            y = y.float()

            # ✅ Use normalizer here
            x = normalizer.transform(x)

            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")

    version = get_next_version()

    model_path = f"ml/models/model_v{version}.pth"
    torch.save(model.state_dict(), model_path)

    # Save metadata
    metadata = {
        "latest_version": version,
        "models": []
    }

    metadata_path = "ml/models/metadata.json"

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    metadata["latest_version"] = version
    metadata["models"].append({
        "version": version,
        "loss": total_loss / len(loader)
    })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Model v{version} saved")


if __name__ == "__main__":
    train()