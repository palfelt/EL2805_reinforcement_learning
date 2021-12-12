import numpy as np
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

# Load model
try:
    model = torch.load('F:\VScode projects\EL2805_reinforcement_learning\lab2\problem1\model5-600epc.pt')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-5.pth not found!')
    exit(-1)


num_y = 200
num_w = 200
y = np.linspace(start=0., stop=1.5, num=num_y)
w = np.linspace(start=-np.pi, stop=np.pi, num=num_w)
yv, wv = np.meshgrid(y, w)
q_values = np.zeros(shape=(num_y, num_w))
a_values = np.zeros(shape=(num_y, num_w))

for iy, ix in np.ndindex(q_values.shape):
    state_tensor = torch.tensor([0, yv[iy, ix], 0, 0, wv[iy, ix], 0, 0, 0], dtype=torch.float32)
    q_max, a_max = torch.max(model(state_tensor), axis=0)
    q_values[iy, ix] = q_max.item()
    a_values[iy, ix] = a_max.item()

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(yv, wv, q_values, cmap='hot')
ax.set_xlabel('$y$')
ax.set_ylabel('$\omega$')
ax.set_zlabel('$max_a Q$')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(yv, wv, a_values, cmap='hot')
ax.set_xlabel('$y$')
ax.set_ylabel('$\omega$')
ax.set_zlabel('$argmax_a Q$')
plt.show()

# s = [x, y, x_dot, y_dot, angle, w, left_contact, right_contact]