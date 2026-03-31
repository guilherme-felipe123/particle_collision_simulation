import random

class Detector:
    def __init__(self, noise_level=0.05, detection_prob=0.9):
        self.noise_level = noise_level
        self.detection_prob = detection_prob

    def add_noise(self, value):
        noise = random.uniform(-self.noise_level, self.noise_level)
        return value * (1 + noise)

    def observe_particle(self, particle):
        # simulate missed detection
        if random.random() > self.detection_prob:
            return None

        return {
            "px": self.add_noise(particle["px"]),
            "py": self.add_noise(particle["py"]),
            "energy": self.add_noise(particle["energy"])
        }

    def observe(self, true_event):
        measured_particles = []

        for p in true_event["particles"]:
            observed = self.observe_particle(p)
            if observed is not None:
                measured_particles.append(observed)

        return {
            "n_detected": len(measured_particles),
            "particles": measured_particles
        }