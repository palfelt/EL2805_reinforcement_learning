from matplotlib.patches import FancyArrow
import numpy as np
#import minotaur_maze_VI as mz 
#import minotaur_maze_DP as mz
import minotaur_maze_Q as mz
#import minotaur_maze_SARSA as mz
import matplotlib.pyplot as plt

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


# method = 'SARSA'
# start  = (0,0,6,5,1,0)
# n_exits = 0
# policy, V_episode = mz.sarsa(env, eps=0.1, n_episodes=50000, gamma=0.95)
# # plt.plot(V_episode)
# # plt.show()
# # print(V_episode[-1])
# n_sim = 500000
# for i in range(1, n_sim + 1):

#     if i % 5000 == 0: 
#         print('i = {}'.format(i))

#     path = env.simulate(start, policy, method)
#     if path[-1][0:2] == (6, 5): # if the player exited the maze
#         n_exits += 1

# print(path)
# prob = n_exits / i
# print("Probability of exiting the maze: " + str(prob))


method = 'Q-learning'
start  = (0,0,6,5,1,0)
n_exits = 0
policy, V_episode = mz.q_learning(env, eps=0.25, n_episodes=50000, gamma=0.95, alp=2/3, fac=0)
# print(V_episode[-1])
# policy1, V_episode1 = mz.q_learning(env, eps=0.2, n_episodes=50000, gamma=0.95, alp=2/3, fac=1)
# policy2, V_episode2 = mz.q_learning(env, eps=0.2, n_episodes=50000, gamma=0.95, alp=2/3, fac=4)
# p1, = plt.plot(V_episode)
# p2, = plt.plot(V_episode1)
# p3, = plt.plot(V_episode2)
# l1 = plt.legend([p1, p2, p3], ["Q = 0", "Q = 1", "Q = 4"], loc='upper right')
# plt.gca().add_artist(l1)
# plt.show()
n_sim = 500000
for i in range(1, n_sim + 1):

    if i % 5000 == 0: 
        print('i = {}'.format(i))

    path = env.simulate(start, policy, method)
    if path[-1][0:2] == (6, 5): # if the player exited the maze
        n_exits += 1

print(path)
prob = n_exits / i
print("Probability of exiting the maze: " + str(prob))