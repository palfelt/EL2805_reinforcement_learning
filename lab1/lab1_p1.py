import numpy as np
import minotaur_maze as mz 

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
# with the convention 
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze

# mz.draw_maze(maze)

# Create an environment maze
env = mz.Maze(maze)
# env.show()

# # Finite horizon
# horizon = 20
# # Solve the MDP problem with dynamic programming 
# V, policy, map= mz.dynamic_programming(env,horizon);
# # print(policy[:,0])

# show_action = dict()
# val = list()
# k = 0
# for i in range(maze.shape[0]):
#     for j in range(maze.shape[1]):
#         if maze[i,j] != 1:
#         # show_action[k] = (i, j, 6, 5)
#             val.append(map[(i,j, 6, 5)]) 
# print(policy[val, 0])

# Simulate the shortest path starting from position A
# method = 'DynProg';
# start  = (0,0,6,5);
# path = env.simulate(start, policy, method);

# Show the shortest path 
# mz.animate_solution(maze, path)

method = 'DynProg';
start  = (0,0,6,5);
prob = np.zeros(shape=16)
n_sim = 100000
m = 0
for T in range(15, 31):
    print("T: " + str(T))
    count = 0
    V, policy, map= mz.dynamic_programming(env, T);
    for i in range(n_sim):
        path = env.simulate(start, policy, method);
        if path[-1][0:2] == (6, 5):
            count += 1
    prob[m] = count / n_sim
    print(prob[m])
    m += 1
print(prob)
# np.savetxt('exit_prob.txt', prob)


# # value iteration:
# p = 1/31 # mean to die at T = 30
# sum = 0
# for k in range(100):
#     geometric_prob = (1 - p) ** (k - 1) * p
#     sample = np.random.rand()
#     if sample <= geometric_prob:
#         T = k
#         break
#     print(sample)

    

