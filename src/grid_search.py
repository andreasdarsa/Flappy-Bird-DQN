import itertools
import numpy as np

from flappy_bird import FlappyBirdEnv
from agent import Agent


def normalize(state):
    return state / np.array([500, 10, 400, 500])


param_grid = {
    "lr": [1e-3, 5e-4],
    "epsilon_decay": [0.995, 0.998],
    "gamma": [0.99, 0.95],
    "distance_weight": [0.01, 0.02]
}


def run_experiment(params):
    env = FlappyBirdEnv(render_mode=False, distance_weight=params["distance_weight"])
    agent = Agent(
        lr=params["lr"],
        gamma=params["gamma"],
        epsilon_decay=params["epsilon_decay"]
    )

    episodes = 500
    scores = []

    for ep in range(episodes):
        state = normalize(env.reset())
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = normalize(next_state)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state

        agent.epsilon = max(0.05, agent.epsilon * agent.epsilon_decay)
        scores.append(env.score)

    return np.mean(scores[-50:])  # average of last episodes


def grid_search():
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    results = []

    for values in combinations:
        params = dict(zip(keys, values))

        print("Running:", params)
        score = run_experiment(params)

        results.append((params, score))

        print("Score:", score)
        print("-" * 40)

    # sort results
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n=== BEST CONFIGS ===")
    for r in results[:5]:
        print(r)


if __name__ == "__main__":
    grid_search()
