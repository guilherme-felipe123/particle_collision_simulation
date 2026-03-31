import math
import torch
import pygame
import random
import itertools

from particle import Particle
from genarate_data import main
from physics import check_collision
from detector.detector import Detector
from data.dataset import DatasetWriter
from ml.deepset_model import DeepSetModel
from ml.preprocess import extract_features



def get_speed(p):
    return math.hypot(p.vx, p.vy)

def get_angle(p1, p2):
    return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))


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

writer = DatasetWriter()

#main()

model = DeepSetModel()
model.load_state_dict(torch.load("/app/ml/model.pth"))
model.eval()
print("Model loaded!")
print(next(model.parameters()).mean())

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

                # compute event
                v1 = get_speed(p1)
                v2 = get_speed(p2)
                angle = get_angle(p1, p2)

                particles = []

                # create explosion
                true_particles = []

                n_particles = random.randint(4, 12)

                for _ in range(n_particles):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 4)

                    vx = math.cos(angle) * speed
                    vy = math.sin(angle) * speed

                    particle = Particle(
                        x=400,
                        y=300,
                        vx=vx,
                        vy=vy,
                        radius=4
                    )

                    particles.append(particle)

                    # ✅ collect physics data
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

                # prepare input for model
                measured_particles = measured_event["particles"]

                #x = extract_features(measured_event)

                # convert to tensor
                x = [[p["px"], p["py"], p["energy"]] for p in measured_particles]
                x_tensor = torch.tensor(x, dtype=torch.float).unsqueeze(0)  # (1, n, 3)



                with torch.no_grad():
                    pred = model(x_tensor)

                print("Model input:", x)
                print("Prediction raw:", pred.item())
                
                pred_particles = pred.item() * 20
                pred_particles = max(0, min(pred_particles, 20))

                writer.save_event(true_event, measured_event)

                true_n = true_event["n_particles"]

                print(f"TRUE: {true_n}")
                print(f"PREDICTED: {pred_particles:.2f}")

                

                #print("TRUE:", true_event)
                #print("MEASURED:", measured_event)

                break

    error = abs(true_n - pred_particles)

    text = font.render(
        f"True: {true_n} | Pred: {pred_particles:.2f} | Err: {error:.2f}",
        True,
        (255, 255, 255)
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