import numpy as np

def extract_features(measured_event):
    particles = measured_event["particles"]

    if len(particles) == 0:
        return [0]*6

    energies = np.array([p["energy"] for p in particles])
    px = np.array([p["px"] for p in particles])
    py = np.array([p["py"] for p in particles])

    n = len(particles)

    features = [
        n / 20,                          # normalize count
        energies.sum() / 50,            # normalize sum
        energies.mean() / 5,
        energies.max() / 5,
        energies.std() / 5,
        np.sqrt(px.sum()**2 + py.sum()**2) / 50
    ]

    return features