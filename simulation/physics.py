import math

def check_collision(p1, p2):
    dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
    return dist < (p1.radius + p2.radius)

def resolve_collision(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = math.hypot(dx, dy)

    if distance == 0:
        return

    # Normalize collision vector
    nx = dx / distance
    ny = dy / distance

    # Relative velocity
    dvx = p1.vx - p2.vx
    dvy = p1.vy - p2.vy

    # Velocity along normal
    vn = dvx * nx + dvy * ny

    # If they are moving apart, do nothing
    if vn > 0:
        return

    # Elastic collision (equal mass)
    impulse = -2 * vn / 2

    p1.vx += impulse * nx
    p1.vy += impulse * ny
    p2.vx -= impulse * nx
    p2.vy -= impulse * ny

def separate_particles(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = math.hypot(dx, dy)

    overlap = (p1.radius + p2.radius) - distance

    if overlap > 0:
        correction = overlap / 2
        nx = dx / distance
        ny = dy / distance

        p1.x -= nx * correction
        p1.y -= ny * correction
        p2.x += nx * correction
        p2.y += ny * correction