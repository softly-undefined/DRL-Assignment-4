# going for the DDPG algorithm again
import gymnasium as gym
import numpy as np
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from dmc import make_dmc_env

class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000, batch_size=128):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transition = namedtuple('Transition', ['s', 'a', 'r', 's2', 'd'])

    def add(self, s, a, r, s2, d):
        self.buffer.append(self.transition(s, a, r, s2, d))

    def sample(self):
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[idx] for idx in batch]
        s = torch.FloatTensor([t.s for t in batch])
        a = torch.FloatTensor([t.a for t in batch])
        r = torch.FloatTensor([t.r for t in batch]).unsqueeze(1)
        s2 = torch.FloatTensor([t.s2 for t in batch])
        d = torch.FloatTensor([t.d for t in batch]).unsqueeze(1)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

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

    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)#, nn.Tanh()
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        #self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)


        #hyperparams
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.noise_std = 0.1
        
        self.state_dim = 5
        self.action_dim = 1
        self.max_action = 1.0

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 0.001)

        self.buffer = ReplayBuffer(buffer_size=1_000_000, batch_size=self.batch_size)



    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]

        noise = np.random.normal(0, self.noise_std, size=self.action_dim)
        action = np.clip(action + noise, -self.max_action, self.max_action)

        return action


    def step(self, s, a, r, s2, d):
        self.buffer.add(s, a, r, s2, d)

        if len(self.buffer) >= self.batch_size:
            self.update()

    
    def update(self):
        s, a, r, s2, d = self.buffer.sample()

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            q_target = r + (1 - d) * self.gamma * q2

        q = self.critic(s, a)
        critic_loss = nn.MSELoss()(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)
    
    def load(self, path):
        self.actor.load_state_dict(torch.load(path))

def make_env():
	# Create environment with state observations
	env_name = "cartpole-balance"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env



def main():
    env = make_env()
    agent = Agent()
    max_episodes = 500
    max_steps = 200

    results = []

    for ep in tqdm(range(1, max_episodes + 1)):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        results.append((ep, total_reward))
        if ep % 5 == 0:
            print(f"episode {ep} - reward {total_reward}")

    agent.save("model.pth")

    with open('training_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['episode', 'reward'])
        writer.writerows(results)

    print("Training complete and results saved to csv!")

if __name__ == "__main__":
    main()

