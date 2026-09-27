from flappy_bird import FlappyBirdEnv
from agent import Agent

import numpy as np
import torch
import matplotlib.pyplot as plt


BEST_CONFIG = {
    "lr": 5e-4,
    "epsilon_decay": 0.998,
    "gamma": 0.95,
    "distance_weight": 0.01,
    "episodes": 2000
}


def normalize(state):
    return state / np.array([500, 10, 400, 500], dtype=np.float32)


def moving_average(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def train():
    env = FlappyBirdEnv(
        render_mode=False,
        distance_weight=BEST_CONFIG["distance_weight"]
    )

    agent = Agent(
        lr=BEST_CONFIG["lr"],
        gamma=BEST_CONFIG["gamma"],
        epsilon_decay=BEST_CONFIG["epsilon_decay"]
    )

    scores = []
    rewards = []
    losses = []

    best_score = 0

    for ep in range(BEST_CONFIG["episodes"]):
        state = normalize(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = normalize(next_state)

            agent.buffer.push(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        agent.update_epsilon(epsilon_min=0.05)

        if ep % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        scores.append(env.score)
        rewards.append(total_reward)

        if env.score > best_score:
            best_score = env.score
            torch.save(agent.model.state_dict(), "best_flappy_model.pth")

        if ep % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_reward = np.mean(rewards[-50:])
            print(
                f"Episode {ep} | "
                f"Score: {env.score} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    torch.save(agent.model.state_dict(), "final_flappy_model.pth")

    plt.figure()
    plt.plot(scores, label="Score")
    plt.plot(moving_average(scores, 50), label="Moving Average (50)")
    plt.title("Flappy Bird DQN - Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("training_scores.png")
    plt.show()

    plt.figure()
    plt.plot(rewards, label="Reward")
    plt.plot(moving_average(rewards, 50), label="Moving Average (50)")
    plt.title("Flappy Bird DQN - Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("training_rewards.png")
    plt.show()

    print("Training completed.")
    print(f"Best score achieved: {best_score}")
    print(f"Final average score over last 50 episodes: {np.mean(scores[-50:]):.2f}")


if __name__ == "__main__":
    train()