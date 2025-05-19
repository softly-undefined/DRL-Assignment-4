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
import json
import time

CHECKPOINT_EVERY  = 500  
LOG_INTERVAL      = 10  
MIN_REPLAY_SIZE   = 10000

# === DEVICE SETUP ===
device = torch.device("cuda:1" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

print(f"Running on {device}")

class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000, batch_size=256):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transition = namedtuple('Transition', ['s', 'a', 'r', 's2', 'd'])

    def add(self, s, a, r, s2, d):
        self.buffer.append(self.transition(s, a, r, s2, d))

    def sample(self):
        idxs = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]

        states      = np.stack([t.s  for t in batch], axis=0)
        actions     = np.stack([t.a  for t in batch], axis=0)
        rewards     = np.stack([t.r  for t in batch], axis=0).reshape(-1,1)
        next_states = np.stack([t.s2 for t in batch], axis=0)
        dones       = np.stack([t.d  for t in batch], axis=0).reshape(-1,1)

        s  = torch.from_numpy(states).float().to(device)
        a  = torch.from_numpy(actions).float().to(device)
        r  = torch.from_numpy(rewards).float().to(device)
        s2 = torch.from_numpy(next_states).float().to(device)
        d  = torch.from_numpy(dones).float().to(device)

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
        self.noise_std = 0.2
        
        self.state_dim = 67
        self.action_dim = 21
        self.max_action = 1.0

        self.actor        = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.critic       = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target= Critic(self.state_dim, self.action_dim).to(device)


        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 2.5e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 2.5e-4)

        self.actor_scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.9992)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9992)

        self.buffer = ReplayBuffer(buffer_size=1_000_000, batch_size=self.batch_size)



    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        noise  = np.random.normal(0, self.noise_std, size=self.action_dim)
        return np.clip(action + noise, -self.max_action, self.max_action)



    def step(self, s, a, r, s2, d):
        self.buffer.add(s, a, r, s2, d)

        if len(self.buffer) >= MIN_REPLAY_SIZE:
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
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env


def main():
    env   = make_env()
    agent = Agent()
    results = []

    max_episodes = 10000
    for ep in tqdm(range(1, max_episodes+1), desc="DDPG training"):
        start_time = time.time()
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if len(agent.buffer) < MIN_REPLAY_SIZE:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)

            state        = next_state
            total_reward += reward
            steps        += 1

        elapsed = time.time() - start_time
        results.append((ep, total_reward, elapsed, steps))

        if ep % LOG_INTERVAL == 0:
            last_rewards = [r for (_, r, _, _) in results[-LOG_INTERVAL:]]
            stats = {
                "episode":            ep,
                "reward":             total_reward,
                "avg_reward_last_10": sum(last_rewards)/len(last_rewards),
                "max_reward_so_far":  max(r for (_, r, _, _) in results),
                "elapsed_time":       elapsed,
                "buffer_size":        len(agent.buffer),
                "steps":              steps
            }
            print(json.dumps(stats), flush=True)

        if ep % CHECKPOINT_EVERY == 0 and ep > 0:
            with open('ddpg_results.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode','reward','time','steps'])
                writer.writerows(results)
            agent.save(f'ddpg_actor_ep{ep}.pth')
            print(f"Checkpointed at episode {ep}")

    # final save
    agent.save("ddpg_actor_final.pth")
    with open('ddpg_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode','reward','time','steps'])
        writer.writerows(results)

    print("Training complete and results saved.")

if __name__ == "__main__":
    main()
