from matplotlib.patches import FancyArrow
import numpy as np

import minotaur_maze_SARSA as mz
import matplotlib.pyplot as plt

"""Group members:
Oscar Palfelt, 980613-7874
Weiqi Xu, 990426-2939
"""

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

# Create an environment maze
env = mz.Maze(maze)

method = 'SARSA'
start  = (0,0,6,5,1,0)
n_exits = 0
policy, V_episode = mz.sarsa(env, eps=0.1, n_episodes=50000, gamma=0.95)
n_sim = 50000
for i in range(1, n_sim + 1):

    if i % 5000 == 0: 
        print('i = {}'.format(i))

    path = env.simulate(start, policy, method)
    if path[-1][0:2] == (6, 5): # if the player exited the maze
        n_exits += 1

prob = n_exits / i
print("Probability of exiting the maze: " + str(prob))


# plt.plot(V_episode)
# plt.show()
# print(V_episode[-1])