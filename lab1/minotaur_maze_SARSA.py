import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
from numpy.random.mtrand import rand

# Implemented methods
methods = ['SARSA'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 1
    KEY_REWARD = 2
    IMPOSSIBLE_REWARD = -100
    CAUGHT_REWARD = -100
    DEAD_REWARD = 0

    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
    
    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1, 0);
        actions[self.MOVE_DOWN]  = (1, 0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    for m in range(self.maze.shape[0]):
                        for n in range(self.maze.shape[1]):
                            for alive in range(2):
                                for have_key in range(2):
                                    states[s] = (i, j, m, n, alive, have_key);
                                    map[(i,j, m, n, alive, have_key)] = s;
                                    s += 1;

        return states, map

    def move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # if caught or exited
        is_caught = self.states[state][0:2] == self.states[state][2:4]
        exited_maze = self.states[state][0:2] == (6,5)
        if exited_maze or is_caught:
            return state

        # if dies from poison
        if np.random.rand() < 1/51 or not self.states[state][4]:
            return self.map[(self.states[state][0], self.states[state][1], self.states[state][2], self.states[state][3], 0, self.states[state][5])]

        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        minotaur_maze_walls = True
        while minotaur_maze_walls:
            if np.random.rand() < 0.65:
                a = np.random.randint(low=0, high=4)
                b = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            else:
                diff = np.array(self.states[state][0:2]) - np.array(self.states[state][2:4])
                move_direction = np.sign(diff)
                b = []
                dirc = np.zeros(2)
                if (move_direction == np.array([0, 0])).all():
                    a = np.random.randint(low=0, high=4)
                    b = [[-1, 0], [1, 0], [0, 1], [0, -1]]
                else:
                    for i, v in enumerate(move_direction):
                        if v == 0:
                            pass
                        else:
                            dirc[i] = v
                            dirc[1-i] = 0
                            b.append(dirc)
                    a = np.random.randint(low=0, high=len(b))

            next_minotaur_row = self.states[state][2] + b[a][0]
            next_minotaur_col = self.states[state][3] + b[a][1]

            
            minotaur_maze_walls = (next_minotaur_row == -1) or (next_minotaur_row == self.maze.shape[0]) or \
                                (next_minotaur_col == -1) or (next_minotaur_col == self.maze.shape[1])

        minotaur_row = self.states[state][2] + b[a][0]
        minotaur_col = self.states[state][3] + b[a][1]

        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);

        if (row, col) == (0,7) or self.states[state][5] == 1:
            have_key = 1
        else:
            have_key = 0

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls: # if move in wall
            return self.map[(self.states[state][0], self.states[state][1], minotaur_row, minotaur_col, 1, have_key)]
        else:
            return self.map[(row, col, minotaur_row, minotaur_col, 1, have_key)];

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.move(s,a);

                minotaur_in_corner = ((self.states[next_s][2] == 0) and (self.states[next_s][3] == self.maze.shape[1] - 1)) or \
                                     ((self.states[next_s][3] == 0) and (self.states[next_s][2] == self.maze.shape[0] - 1)) or \
                                     ((self.states[next_s][3] == 0) and (self.states[next_s][2] == 0)) or \
                                     ((self.states[next_s][2] == self.maze.shape[0] - 1) and (self.states[next_s][3] == self.maze.shape[1] - 1))

                minotaur_in_edge = (self.states[next_s][2] == 0) or (self.states[next_s][2] == self.maze.shape[0] - 1) or \
                                (self.states[next_s][3] == 0) or (self.states[next_s][3] == self.maze.shape[1] - 1)

                minotaur_in_sameline = self.states[s][0] == self.states[s][2] or self.states[s][1] == self.states[s][3]

                diff = np.sign(np.array(self.states[s][0:2]) - np.array(self.states[s][2:4]))
                minotaur_diff = np.sign(np.array(self.states[next_s][2:4]) - np.array(self.states[s][2:4]))

                is_caught = self.states[s][0:2] == self.states[s][2:4]
                exited_maze = self.states[s][0:2] == (6, 5)

                if is_caught or exited_maze or not self.states[s][4]:
                    transition_probabilities[next_s, s, a] = 1
                elif minotaur_in_corner:
                    if minotaur_in_sameline:
                        if (minotaur_diff == diff).all():
                            transition_probabilities[next_s, s, a] = 0.675;
                        else:
                            transition_probabilities[next_s, s, a] = 0.325;
                    else:
                        transition_probabilities[next_s, s, a] = 0.5;
                elif minotaur_in_edge:
                    if minotaur_in_sameline:
                        if (minotaur_diff == diff).all():
                            transition_probabilities[next_s, s, a] = 0.35 + 0.65/3;
                        else:
                            transition_probabilities[next_s, s, a] = 0.65/3;
                    else:
                        if minotaur_diff[0] == diff[0] or minotaur_diff[1] == diff[1]:
                            transition_probabilities[next_s, s, a] = 0.35/2 + 0.65/3;
                        else:
                            transition_probabilities[next_s, s, a] = 0.65/3;
                else:
                    if minotaur_in_sameline:
                        if (minotaur_diff == diff).all():
                            transition_probabilities[next_s, s, a] = 0.35 + 0.65/4;
                        else:
                            transition_probabilities[next_s, s, a] = 0.65/4;
                    else:
                        if minotaur_diff[0] == diff[0] or minotaur_diff[1] == diff[1]:
                            transition_probabilities[next_s, s, a] = 0.35/2 + 0.65/4;
                        else:
                            transition_probabilities[next_s, s, a] = 0.65/4;

        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.move(s,a);

                is_caught = self.states[s][0:2] == self.states[s][2:4]
                exited_maze = self.states[s][0:2] == (6, 5)

                # # Reward for being dead
                if not self.states[next_s][4]:
                    rewards[s,a] = self.DEAD_REWARD
                # Reward for hitting a wall
                elif self.states[s][0:2] == self.states[next_s][0:2] and a != self.STAY and not is_caught and not exited_maze:
                    rewards[s,a] = self.IMPOSSIBLE_REWARD;
                # Reward for being caught by minotaur
                elif is_caught:
                    rewards[s,a] = self.CAUGHT_REWARD;
                # Reward for picking up key
                elif self.states[s][5] == 0 and self.states[next_s][5] == 1:
                    rewards[s,a] = self.KEY_REWARD;
                # Reward for reaching the exit
                elif exited_maze and self.states[s][5] == 1:
                    rewards[s,a] = self.GOAL_REWARD;
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s,a] = self.STEP_REWARD;

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();

        if method == 'SARSA':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state, caught or dead
            exit_maze = self.states[s][0:2] == (6, 5)
            caught = self.states[s][0:2] == self.states[next_s][2:4]
            dead = self.states[s][4] == 0
            while not exit_maze and not caught and not dead:

                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;

                exit_maze = self.states[next_s][0:2] == (6, 5)
                caught = self.states[next_s][0:2] == self.states[next_s][2:4]
                dead = self.states[next_s][4] == 0
            

        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def sarsa(env, eps=0.2, n_episodes=50000, gamma=0.95):
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V_episode   = np.zeros(n_episodes)
    V   = np.zeros(n_states)
    #BV  = np.zeros(n_states)
    Q = 0 * np.ones((n_states, n_actions))
    n = np.zeros((n_states, n_actions))
    delta = 0.7

    for epispde in range(1, n_episodes+1):
        s = env.map[(0,0,6,5,1,0)] # unsure of the initial state

        exit_maze = env.states[s][0:2] == (6, 5)
        caught = env.states[s][0:2] == env.states[s][2:4]
        dead = env.states[s][4] == 0

        if np.random.rand() < 1/epispde**delta:    # eps - epispde * 0.000004:
            a = np.random.randint(low=0, high=n_actions-1)  # select random action with probability eps
        else:
            a = np.argmax(Q[s], axis=0)
        #BV = np.max(Q, 1)
        # a = np.argmax(Q[s], axis=0)

        t = 0
        while not exit_maze and not caught and not dead:
            t += 1

            next_s = env.move(s, a) 
            if np.random.rand() < 1/epispde**delta: # eps - epispde * 0.000004:
                next_a = np.random.randint(low=0, high=n_actions-1)
            else:
                next_a = np.argmax(Q[next_s], axis=0)

            n[s, a] += 1
            alpha = 1 / n[s, a] ** (2 / 3)
            Q[s, a] += alpha * (r[s, a] + gamma * Q[next_s, next_a] - Q[s, a])
            s = next_s
            a = next_a

            exit_maze = env.states[s][0:2] == (6, 5)
            caught = env.states[s][0:2] == env.states[s][2:4]
            dead = env.states[s][4] == 0

        V = np.max(Q, 1)
        # print("Exited maze: " + str(exit_maze))
        # print("Got caught: " + str(caught))
        # print("Died: " + str(dead))
        V_episode[epispde-1] = V[env.map[(0,0,6,5,1,0)]]

    policy = np.argmax(Q,axis=1);  
    return policy, V_episode


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);
