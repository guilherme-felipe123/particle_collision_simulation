import math
import random

from detector.detector import Detector
from data.dataset import DatasetWriter


def generate_event():
    true_particles = []

    for _ in range(random.randint(5, 10)):  # vary number of particles
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)

        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed

        true_particles.append({
            "px": vx,
            "py": vy,
            "energy": math.hypot(vx, vy)
        })

    true_event = {
        "n_particles": len(true_particles),
        "particles": true_particles
    }

    return true_event


def main():
    detector = Detector(noise_level=0.1)
    writer = DatasetWriter()

    N_EVENTS = 1000

    for i in range(N_EVENTS):
        true_event = generate_event()
        measured_event = detector.observe(true_event)

        writer.save_event(true_event, measured_event)

        if i % 100 == 0:
            print(f"Generated {i} events")


if __name__ == "__main__":
    main()