import pygame

class Particle:
    def __init__(self, x, y, vx, vy, radius=10):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius)

    def bounce_walls(self, width, height):
        if self.x - self.radius <= 0 or self.x + self.radius >= width:
            self.vx *= -1

        if self.y - self.radius <= 0 or self.y + self.radius >= height:
            self.vy *= -1