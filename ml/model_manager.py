import json
import torch


def load_best_model(model, metadata_path="ml/models/metadata.json"):
    with open(metadata_path) as f:
        metadata = json.load(f)

    # find best model (lowest loss)
    best = min(metadata["models"], key=lambda x: x["loss"])

    version = best["version"]
    path = f"ml/models/model_v{version}.pth"

    model.load_state_dict(torch.load(path, map_location="cpu"))

    print(f"✅ Loaded best model: v{version} (loss={best['loss']:.4f})")

    return model