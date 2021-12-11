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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
from math import gamma
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent
import torch.nn as nn
from collections import deque
from torch.nn.modules import loss
import torch.optim as optim
from DDPG_soft_updates import soft_updates

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=30000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return self.buffer.maxlen

    def sample_batch(self, n):
        if n > len(self.buffer):
            print("Too few elements in buffer")
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), n, replace=False)]

        # return a tuple of five lists of len n
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

class CriticNetwork(nn.Module):
    def __init__(self , d, m, K):
        super (). __init__ ()
        hidden_neurons_1 = 400 # Number of hidden neurons
        hidden_neurons_2 = 200
        # The input dimensionality of the network should be equal to the
        # dimensionality of the state
        self.input_state_layer = nn.Linear(d, hidden_neurons_1)
        self.input_activation = nn.ReLU()
        # The dimensionality of the input of the next layer should
        # be equal to the dimensionality of the hidden layer
        # + dimensionality of the actions
        # The output should be equal to 1 since we are computing
        # just one Q value
        self.hidden_layer = nn.Linear(hidden_neurons_1 + m, hidden_neurons_2)
        self.hidden_activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_neurons_2, 1)
        # Use the Tanh activation to bound the output between -1 and 1
        self.output_activation = nn.Tanh()
        self.K = K

    def forward(self , s, a):
        # Computation of Q(s,a)
        # Compute the hidden layer
        h1_state = self.input_activation(self.input_state_layer(s))
        # Concatenate output of the hidden layer with the action along
        # the dimensionality of the data
        hidden = torch.cat([h1_state , a], dim=1)
        h2_state = self.hidden_activation(self.hidden_layer(hidden))
        # Compute ouput
        out = self.output_layer(h2_state)
        # Multiply the output of the activation function by K to get
        # that the output is between -K and K
        return self.K * self.output_activation(out)

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

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
dim_state = len(env.observation_space.high)      # dimensionality of the state
N_episodes = 300                                 # Number of episodes to run for training
discount_factor = 0.99                           # Value of gamma
n_ep_running_average = 50                        # Running average of 50 episodes
m = len(env.action_space.high)                   # dimensionality of the action
batch_size = 64
Actor = ActorNetwork(input_size=dim_state, output_size=m)
Target_Actor = ActorNetwork(input_size=dim_state, output_size=m)
Critic = CriticNetwork(dim_state, m, K=1)
Target_Critic = CriticNetwork(dim_state, m, K=1)
actor_optimizer = optim.Adam(Actor.parameters(), lr=5e-5)
critic_optimizer = optim.Adam(Critic.parameters(), lr=5e-4)

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
rnd_agent = RandomAgent(m)

buffer = ExperienceReplayBuffer(maximum_length=30000)
for i in range(len(buffer)):
    rnd_state = env.reset()
    rnd_action = rnd_agent.forward(rnd_state)
    next_state, reward, done, _ = env.step(rnd_action)
    rnd_exp = (rnd_state, rnd_action, reward, next_state, done)
    buffer.append(rnd_exp)


# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    n = 0
    while not done:
        # Take a random action
        state_tensor = torch.tensor(state,
            requires_grad=False, dtype=torch.float32)
        action = Actor(state_tensor) + n
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action.detach().numpy())
        exp = (state, action.detach().numpy(), reward, next_state, done)
        buffer.append(exp)

        states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)

        actions_tensor = torch.tensor(actions,
                            requires_grad=True, dtype=torch.float32)

        next_states_tensor = torch.tensor(next_states,
                            requires_grad=False, dtype=torch.float32) # used for target action/critic network

        states_tensor = torch.tensor(states,
                            requires_grad=True, dtype=torch.float32) # critic network for loss calculation
        
        rewards_tensor = torch.tensor(rewards, requires_grad=True, dtype=torch.float32)

        next_actions_tensor = Target_Actor(next_states_tensor)

        Q = discount_factor * (1 - torch.tensor(dones, requires_grad=True, dtype=torch.float32)) * torch.flatten(Target_Critic(next_states_tensor, next_actions_tensor))
        y = rewards_tensor + Q
        
        critic_optimizer.zero_grad()
        state_action_values = Critic(states_tensor, actions_tensor)
        loss = nn.functional.mse_loss(state_action_values.flatten(), y)
        loss.backward()
        nn.utils.clip_grad_norm_(Critic.parameters(), 1.0)
        critic_optimizer.step()
        
        if t % 2 == 0:
            actor_optimizer.zero_grad()
            J = Critic(states_tensor, Actor(states_tensor)).sum()
            J.backward()

            nn.utils.clip_grad_norm_(Actor.parameters(), 1.0)
            actor_optimizer.step()
            Target_Actor = soft_updates(Actor, Target_Actor, 1e-3)
            Target_Critic = soft_updates(Critic, Target_Critic, 1e-3)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1
        w = np.random.normal(0, 0.2)
        n = -0.15 * n + w

    # Append episode reward
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
