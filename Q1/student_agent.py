import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.net(x) * self.max_action


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        #self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

        self.actor = Actor(state_dim=3, action_dim=1, max_action=2.0)

        self.actor.load_state_dict(torch.load("model.pth", map_location="cpu"))
        self.actor.eval()


    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action