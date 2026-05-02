import math
import time
import torch
import pygame
import random
import requests
import itertools

from particle import Particle
from physics import check_collision
from detector.detector import Detector
from ml.deepset_model import DeepSetModel
from ml.preprocess import extract_features


def get_prediction(particles):
    url = "http://localhost:8000/predict"

    try:
        start = time.time()

        response = requests.post(url, json={
            "particles": particles
        }, timeout=2)

        latency = time.time() - start

        data = response.json()

        if "predicted_particles" not in data:
            print("API ERROR:", data)
            return 0, latency

        return data["predicted_particles"], latency

    except Exception as e:
        print("Connection error:", e)
        return 0, 0

def get_speed(p):
    return math.hypot(p.vx, p.vy)

def get_angle(p1, p2):
    return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

def normalize(particles):
    return [[p[0]/5, p[1]/5, p[2]/5] for p in particles]

def handle_collision(detector):
    true_particles = []

    n_particles = random.randint(4, 12)
    particles = []

    for _ in range(n_particles):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)

        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed

        particle = Particle(400, 300, vx, vy, radius=4)
        particles.append(particle)

        true_particles.append({
            "px": vx,
            "py": vy,
            "energy": math.hypot(vx, vy)
        })

    true_event = {
        "n_particles": len(true_particles),
        "particles": true_particles
    }

    measured_event = detector.observe(true_event)

    x = [[p["px"], p["py"], p["energy"]] for p in measured_event["particles"]]

    return particles, true_event, x

def predict_and_log(x, log_counter):
    prediction, latency = get_prediction(x)

    if log_counter % 30 == 0:
        with open("logs.txt", "a") as f:
            f.write(f"{prediction},{latency}\n")

    return prediction, latency

pygame.init()

screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

detector = Detector(noise_level=0.1)

particles = [
    Particle(100, 300, 2, 0),
    Particle(700, 300, -2, 0)
]
MAX_PARTICLES = 50
true_n = 0
pred_particles = 0

running = True

was_colliding = False
event_active = False

frame_count = 0

total_error = 0
num_events = 0

baseline = 8  # average number of particles

while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for p in particles:
        p.move()
        p.draw(screen)

    
    collided_pairs = []

    for p1, p2 in itertools.combinations(particles, 2):
        is_colliding = check_collision(p1, p2)
        if is_colliding:
            collided_pairs.append((p1, p2))

    if not event_active:
        for p1, p2 in itertools.combinations(particles, 2):
            if check_collision(p1, p2):

                event_active = True
                frame_count = 0

                particles, true_event, x = handle_collision(detector)

                prediction, latency = predict_and_log(x, frame_count)

                pred_particles = prediction * 20
                pred_particles = max(0, min(pred_particles, 20))

                true_n = true_event["n_particles"]

                print(f"TRUE: {true_n}")
                print(f"Prediction: {pred_particles:.3f} | Latency: {latency:.3f}s")

                break

    error = abs(true_n - pred_particles)
    baseline_error = abs(true_n - baseline)
    status = "Good" if error < baseline_error else "Bad"
    diff = baseline_error - error

    if diff > 2:
        color = (0, 255, 0)      # strong win
    elif diff > 0:
        color = (200, 255, 0)    # slight win
    else:
        color = (255, 0, 0)      # loss
    
    total_error += error
    num_events += 1

    avg_error = total_error / num_events if num_events > 0 else 0

    text = font.render(
        f"True: {true_n} | Pred: {pred_particles:.2f} | Err: {error:.2f} | Avg: {avg_error:.2f} | Base Err: {baseline_error:.2f} | {status}",
        True,
        color
    )

    screen.blit(text, (20, 20))

    if len(particles) > MAX_PARTICLES:
        particles = particles[:MAX_PARTICLES]

    if event_active:
        frame_count += 1

    if event_active and frame_count > 120:  # ~2 seconds
        particles = [
            Particle(100, 300, 2, 0),
            Particle(700, 300, -2, 0)
        ]
        event_active = False
        frame_count = 0

    was_colliding = is_colliding

    pygame.display.flip()
    clock.tick(60)

pygame.quit()