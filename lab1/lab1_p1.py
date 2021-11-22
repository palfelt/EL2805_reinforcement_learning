import numpy as np
#import minotaur_maze_VI as mz 
#import minotaur_maze_DP as mz
#import minotaur_maze_Q as mz
import minotaur_maze_SARSA as mz

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

# method = 'DynProg';
# start  = (0,0,6,5);
# prob = np.zeros(shape=31)
# n_sim = 10000
# for T in range(15, 31):
#     print("T: " + str(T))
#     count = 0
#     V, policy= mz.dynamic_programming(env, T);
#     for i in range(n_sim):
#         path = env.simulate(start, policy, method);
#         if path[-1][0:2] == (6, 5):
#             count += 1
#     prob[T] = count / n_sim
#     print(prob[T])

# np.savetxt('exit_prob_caught_10_goal0.txt', prob)

# --- Value iteration ---

# method = 'ValIter'
# # Discount Factor
# gamma   = 0.95; 
# # Accuracy treshold 
# epsilon = 0.0001;
# start  = (0,0,6,5,1)
# n_sim = 100000
# n_exits = 0
# V, policy = mz.value_iteration(env, gamma, epsilon)
# for i in range(1, n_sim + 1):

#     if i % 10000 == 0: 
#         print('i = {}'.format(i))

#     path = env.simulate(start, policy, method)
#     if path[-1][0:2] == (6, 5): # if the player exited the maze
#         n_exits += 1

# prob = n_exits / i
# print("Probability of exiting the maze: " + str(prob))

# method = 'Q-learning'
# start  = (0,0,6,5,1)
# n_exits = 0
# policy = mz.q_learning(env, eps=0.2, n_episodes=50000, gamma=0.95)
# n_sim = 50000
# for i in range(1, n_sim + 1):

#     if i % 5000 == 0: 
#         print('i = {}'.format(i))

#     path = env.simulate(start, policy, method)
#     if path[-1][0:2] == (6, 5): # if the player exited the maze
#         n_exits += 1

# print(path)
# prob = n_exits / i
# print("Probability of exiting the maze: " + str(prob))


method = 'SARSA'
start  = (0,0,6,5,1)
n_exits = 0
policy = mz.sarsa(env, eps=0.2, n_episodes=50000, gamma=0.95)
n_sim = 50000
for i in range(1, n_sim + 1):

    if i % 5000 == 0: 
        print('i = {}'.format(i))

    path = env.simulate(start, policy, method)
    if path[-1][0:2] == (6, 5): # if the player exited the maze
        n_exits += 1

print(path)
prob = n_exits / i
print("Probability of exiting the maze: " + str(prob))