import torch

def collate_fn(batch):
    xs, ys = zip(*batch)

    max_particles = max(len(x) for x in xs)

    padded = []

    for x in xs:
        tensor = torch.tensor(x, dtype=torch.float)

        if len(x) < max_particles:
            pad = torch.zeros((max_particles - len(x), 3))
            tensor = torch.cat([tensor, pad], dim=0)

        padded.append(tensor)

    return torch.stack(padded), torch.tensor(ys).float().unsqueeze(1)