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
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn as nn
import matplotlib.pyplot as plt

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
CONFIDENCE_PASS = 50

# Reward
episode_reward_list = []  # Used to store episodes reward

num_y = 100
num_w = 100
y = np.linspace(start=0., stop=1.5, num=num_y)
w = np.linspace(start=-np.pi, stop=np.pi, num=num_w)
yv, wv = np.meshgrid(y, w)
q_values = np.zeros(shape=(num_y, num_w))
a_values = np.zeros(shape=(num_y, num_w))

j = 0
for iy, ix in np.ndindex(q_values.shape):
    j+=1
    state_tensor = torch.tensor([0, yv[iy, ix], 0, 0, wv[iy, ix], 0, 0, 0], dtype=torch.float32)
    q_max, a_max = torch.max(model(state_tensor), axis=0)
    q_values[iy, ix] = q_max.item()
    a_values[iy, ix] = a_max.item()

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(yv, wv, q_values, cmap='hot')
plt.show()