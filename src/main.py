import numpy as np
import pygame
from flappy_bird import FlappyBirdEnv

MODE = "human"  # human | random

np.random.seed(42)


def main():
    env = FlappyBirdEnv(render_mode=True)
    state = env.reset()

    running = True

    while running:
        action = 0

        if MODE == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    action = 1

        elif MODE == "random":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            action = np.random.choice([0, 1])

        state, reward, done, _ = env.step(action)
        env.render()

        if done:
            print("Game Over | Score:", env.score)
            pygame.time.delay(800)
            state = env.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
