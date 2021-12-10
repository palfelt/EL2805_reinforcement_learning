# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
from threading import excepthook
from typing import Deque, ForwardRef, Optional
import numpy as np
import gym
from numpy.core.numeric import indices, tensordot
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Linear
from tqdm import trange
from DQN_agent import RandomAgent
from collections import deque

# use the code from [Alessio Russo - alessior@kth.se] in p3, LAB0
class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=10000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return self.buffer.maxlen

    def sample_batch(self, n):
        if n > len(self.buffer):
            print("Too few elements in buffer")
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), n - 1, replace=False)]

        #Combined experience replay
        batch.append(self.buffer[-1])

        # return a tuple of five lists of len n
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

        # self.value = nn.Sequential(
        #     nn.Linear(input_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )

        # self.advantage = nn.Sequential(
        #     nn.Linear(input_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, output_size)
        # )


    def forward(self, x):
        return self.network(x)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 500                             # Number of episodes
discount_factor = 0.995                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
rnd_agent = RandomAgent(n_actions)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
cdevice = torch.device("cpu" if torch.cuda.is_available() else "cpu")
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
network = NeuralNetwork(input_size=dim_state, output_size=n_actions)
network.to(cdevice)
optimizer = optim.Adam(network.parameters(), lr=1e-4)
target_network = NeuralNetwork(input_size=dim_state, output_size=n_actions)
target_network.to(cdevice)
batch_size = 64
epsilon_max = 0.99
epsilon_min = 0.1

### Initialization
buffer = ExperienceReplayBuffer(maximum_length=15000)
for i in range(len(buffer)):
    rnd_state = env.reset()
    rnd_action = rnd_agent.forward(rnd_state)
    next_state, reward, done, _ = env.step(rnd_action)
    rnd_exp = (rnd_state, rnd_action, reward, next_state, done)
    buffer.append(rnd_exp)

C = int(len(buffer) / batch_size)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0

    # epsilon decay
    epsilon = np.max([epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * (i - 1) / (0.9 * N_episodes)])
    while not done:

        state_tensor = torch.tensor(state,
            requires_grad=True, dtype=torch.float32, device=cdevice)

        # Take action epsilon-greedily
        if np.random.rand() < epsilon:
            action = rnd_agent.forward(state)
        else:
            state_action_values = network(state_tensor)
            val, action = state_action_values.max(0)
            action = action.item()

        # Get next state, reward and done, append experience to buffer
        next_state, reward, done, _ = env.step(action)
        exp = (state, action, reward, next_state, done)
        buffer.append(exp)

        # sample batch_size experiences from the buffer randomly
        states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
  
        # compute target values
        next_states_tensor = torch.tensor(next_states,
                            requires_grad=True, dtype=torch.float32, device=cdevice)
        states_tensor = torch.tensor(states,
                            requires_grad=True, dtype=torch.float32, device=cdevice)
        next_state_values, _ = target_network(next_states_tensor).max(1)
        y = torch.tensor(rewards, requires_grad=True, dtype=torch.float32, device=cdevice) + discount_factor * (1 - torch.tensor(dones, requires_grad=True, dtype=torch.float32, device=cdevice)) * next_state_values

        # update main network parameters
        optimizer.zero_grad()
        state_action_values = network(states_tensor)[torch.arange(batch_size), list(actions)] # forward pass
        loss = nn.functional.mse_loss(state_action_values, y)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

        # if C steps has passed, set target network equal to main network
        if (t % C) == 0:
            target_network.load_state_dict(network.state_dict())
            
        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    # if running_average(episode_reward_list, n_ep_running_average)[-1] > 150:
    #     break

torch.save(network, 'parameters.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
