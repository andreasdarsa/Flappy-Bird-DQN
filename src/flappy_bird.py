import pygame
import numpy as np


class FlappyBirdEnv:
    def __init__(self, render_mode=False, distance_weight=0.01):
        self.render_mode = render_mode

        self.passed_pipe = None
        self.done = None
        self.pipe_gap_y = None
        self.pipe_x = None
        self.bird_velocity = None
        self.bird_y = None
        self.score = 0

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 500))
            pygame.display.set_caption("Flappy Bird RL")
            self.clock = pygame.time.Clock()

            self.bird_img = pygame.image.load("proj_1/assets/bird.png").convert_alpha()
            self.bird_img = pygame.transform.scale(self.bird_img, (40, 40))

        self.distance_weight = distance_weight

        self.reset()

    def reset(self):
        self.bird_y = 250
        self.bird_velocity = 0

        self.pipe_x = 400
        self.pipe_gap_y = np.random.randint(150, 350)

        self.score = 0
        self.passed_pipe = False

        self.done = False
        return self._get_state()
    
    def step(self, action):
        if action == 1:
            self.bird_velocity = -8

        # physics
        self.bird_velocity += 0.5
        self.bird_y += self.bird_velocity

        # move pipe
        self.pipe_x -= 3

        reward = 1  # survival reward

        if action == 1:
            reward -= 0.1

        # 👉 CHECK if pipe was passed
        if self.pipe_x < 50 and not self.passed_pipe:
            self.passed_pipe = True
            self.score += 1
            reward += 10

        # respawn pipe
        if self.pipe_x < -50:
            self.pipe_x = 400
            self.pipe_gap_y = np.random.randint(150, 350)
            self.passed_pipe = False

        # collision
        if self._check_collision():
            self.done = True
            reward = -100

        # encourage being close to center of gap
        distance = abs(self.bird_y - self.pipe_gap_y)
        reward += -self.distance_weight * distance

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        return np.array([
            self.bird_y,
            self.bird_velocity,
            self.pipe_x,
            self.bird_y - self.pipe_gap_y
        ], dtype=np.float32)

    def _check_collision(self):
        bird_rect = pygame.Rect(30, self.bird_y - 20, 40, 40)

        pipe_width = 50
        gap_size = 180

        top_pipe_rect = pygame.Rect(
            self.pipe_x,
            0,
            pipe_width,
            self.pipe_gap_y - gap_size // 2
        )

        bottom_pipe_rect = pygame.Rect(
            self.pipe_x,
            self.pipe_gap_y + gap_size // 2,
            pipe_width,
            500 - (self.pipe_gap_y + gap_size // 2)
        )

        if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
            return True

        # ground / ceiling
        if self.bird_y < 0 or self.bird_y > 500:
            return True

        return False

    def render(self):
        if not self.render_mode:
            return
        self.screen.fill((135, 206, 235))

        pipe_width = 50
        gap_size = 180

        # --- pipes ---
        pygame.draw.rect(
            self.screen,
            (0, 200, 0),
            (self.pipe_x, 0, pipe_width, self.pipe_gap_y - gap_size // 2)
        )

        bottom_y = self.pipe_gap_y + gap_size // 2
        pygame.draw.rect(
            self.screen,
            (0, 200, 0),
            (self.pipe_x, bottom_y, pipe_width, 500 - bottom_y)
        )

        # --- bird rotation ---
        angle = max(min(-self.bird_velocity * 3, 30), -30)  # tweak sensitivity

        rotated_bird = pygame.transform.rotate(self.bird_img, angle)

        bird_rect = rotated_bird.get_rect(center=(50, int(self.bird_y)))

        self.screen.blit(rotated_bird, bird_rect.topleft)

        # --- score ---
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

