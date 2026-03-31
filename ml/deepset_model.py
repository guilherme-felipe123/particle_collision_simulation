import torch
import torch.nn as nn


class DeepSetModel(nn.Module):
    def __init__(self):
        super().__init__()

        # particle encoder
        self.phi = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # event-level network
        self.rho = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (batch, n_particles, 3)

        x = self.phi(x)          # (batch, n_particles, 16)
        x = x.sum(dim=1)        # (batch, 16) ← aggregation

        return self.rho(x)