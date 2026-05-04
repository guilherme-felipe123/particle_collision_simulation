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


def main(n_events=1000):
    detector = Detector(noise_level=0.1)
    writer = DatasetWriter()

    for i in range(n_events):
        true_event = generate_event()
        measured_event = detector.observe(true_event)

        writer.save_event(true_event, measured_event)

        if i % 100 == 0:
            print(f"Generated {i} events")


if __name__ == "__main__":
    main()