import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Absolute gain range:
        # Kp: [50, 150]
        # Kd: [15, 25]
        self.register_buffer(
            "action_center",
            torch.tensor([30.0, 3.0], dtype=torch.float32)
        )

        self.register_buffer(
            "action_half_range",
            torch.tensor([20.0, 2.0], dtype=torch.float32)
        )

    def forward(self, observation):
        raw_action = self.network(observation)

        # [-1, 1]
        normalized_action = torch.tanh(raw_action)

        # Map to:
        # Kp -> [50, 150]
        # Kd -> [15, 25]
        action = (
            self.action_center
            + self.action_half_range * normalized_action
        )

        return action


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.network(x)