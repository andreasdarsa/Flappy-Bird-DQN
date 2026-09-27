import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from dqn import DQN
from replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon_decay=0.995):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())

        self.buffer = ReplayBuffer()

        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.epsilon = 1.0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def update_epsilon(self, epsilon_min=0.05):
        self.epsilon = max(epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states) \
                .gather(1, next_actions.unsqueeze(1)).squeeze()
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()
