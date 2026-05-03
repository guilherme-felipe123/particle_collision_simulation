import torch
import json


class Normalizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, loader):
        all_particles = []

        for x, _ in loader:
            x = x.float()
            for sample in x:
                all_particles.append(sample)

        all_particles = torch.cat(all_particles, dim=0)

        self.mean = all_particles.mean(dim=0)
        self.std = all_particles.std(dim=0)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist()
            }, f)

    def load(self, path):
        with open(path) as f:
            stats = json.load(f)

        self.mean = torch.tensor(stats["mean"]).float()
        self.std = torch.tensor(stats["std"]).float()