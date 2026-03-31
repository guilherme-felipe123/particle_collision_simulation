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

    model = DeepSetModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(10):
        total_loss = 0

        for x, y in loader:
            x = x.float()
            y = y.float()

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