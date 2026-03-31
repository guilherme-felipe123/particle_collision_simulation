import json
import numpy as np
from torch.utils.data import Dataset


class DatasetWriter:
    def __init__(self, filename="data/events.jsonl"):
        self.filename = filename

    def save_event(self, true_event, measured_event):
        data = {
            "true": true_event,
            "measured": measured_event
        }

        with open(self.filename, "a") as f:
            f.write(json.dumps(data) + "\n")



class CollisionDataset(Dataset):
    def __init__(self, filename):
        self.data = []

        with open(filename, "r") as f:
            for line in f:
                event = json.loads(line)

                self.data.append(event)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        event = self.data[idx]
        
        particles = event["measured"]["particles"]
        y = event["true"]["n_particles"] / 20  # keep normalized

        # convert to list of [px, py, energy]
        x = [[p["px"], p["py"], p["energy"]] for p in particles]

        
        return x, y
