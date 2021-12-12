import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.network(x)

# Load model
try:
    model = torch.load('F:\VScode projects\EL2805_reinforcement_learning\lab2\problem1\model5-600epc.pt')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-5.pth not found!')
    exit(-1)

# Import and initialize Mountain Car Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings

# Reward
network_episode_reward_list = []  # Used to store episodes reward

# Simulate episodes
# Network agent:
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:

        q_values = model(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        next_state, reward, done, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    network_episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

agent = RandomAgent(n_actions=4)
rndagent_episode_reward_list = []

# Random agent:
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:

        # Take random action
        action = agent.forward(state)
        next_state, reward, done, _ = env.step(action)

        total_episode_reward += reward

        state = next_state

    rndagent_episode_reward_list.append(total_episode_reward)

    env.close()

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_EPISODES+1)], network_episode_reward_list, label='Episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes, Q-network')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_EPISODES+1)], rndagent_episode_reward_list, label='Episode reward')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total reward')
ax[1].set_title('Total Reward vs Episodes, Random agent')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.show()
