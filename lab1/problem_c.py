from matplotlib.patches import FancyArrow
import numpy as np
import minotaur_maze_DP as mz

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

method = 'DynProg';
start  = (0,0,6,5);
prob = np.zeros(shape=31)
n_sim = 10000
for T in range(15, 31):
    print("T: " + str(T))
    count = 0
    V, policy= mz.dynamic_programming(env, T);
    for i in range(n_sim):
        path = env.simulate(start, policy, method);
        if path[-1][0:2] == (6, 5):
            count += 1
    prob[T] = count / n_sim
    print(prob[T])

np.savetxt('exit_prob_caught_10_goal0.txt', prob)