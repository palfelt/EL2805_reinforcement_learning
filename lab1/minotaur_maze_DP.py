import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg'];

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
    STEP_REWARD = 0
    GOAL_REWARD = 1 # 5 -> 49.3%, 15 -> 39%
    IMPOSSIBLE_REWARD = -100
    CAUGHT_REWARD = -1


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
                            states[s] = (i, j, m, n);
                            map[(i,j, m, n)] = s;
                            s += 1;

        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # if caught or exited
        is_caught = self.states[state][0:2] == self.states[state][2:4]
        exited_maze = self.states[state][0:2] == (6,5)
        if exited_maze or is_caught:
            return state

        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        minotaur_maze_walls = True
        while minotaur_maze_walls:
            a = np.random.randint(low=0, high=5)
            #b = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            b = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]

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

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return self.map[(self.states[state][0], self.states[state][1], minotaur_row, minotaur_col)]
        else:
            return self.map[(row, col, minotaur_row, minotaur_col)];

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
                next_s = self.__move(s,a);

                minotaur_in_corner = ((self.states[next_s][2] == 0) and (self.states[next_s][3] == self.maze.shape[1] - 1)) or \
                                     ((self.states[next_s][3] == 0) and (self.states[next_s][2] == self.maze.shape[0] - 1)) or \
                                     ((self.states[next_s][3] == 0) and (self.states[next_s][2] == 0)) or \
                                     ((self.states[next_s][2] == self.maze.shape[0] - 1) and (self.states[next_s][3] == self.maze.shape[1] - 1))

                minotaur_in_edge = (self.states[next_s][2] == 0) or (self.states[next_s][2] == self.maze.shape[0] - 1) or \
                                   (self.states[next_s][3] == 0) or (self.states[next_s][3] == self.maze.shape[1] - 1)

                is_caught = self.states[s][0:2] == self.states[s][2:4]
                exited_maze = self.states[s][0:2] == (6, 5)

                if is_caught or exited_maze:
                    transition_probabilities[next_s, s, a] = 1
                elif minotaur_in_corner:
                    transition_probabilities[next_s, s, a] = 1/2;
                elif minotaur_in_edge:
                    transition_probabilities[next_s, s, a] = 1/3;
                else:
                    transition_probabilities[next_s, s, a] = 1/4;

        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s,a);

                is_caught = self.states[s][0:2] == self.states[s][2:4]
                exited_maze = self.states[s][0:2] == (6, 5)
                
                # Reward for hitting a wall
                if self.states[s][0:2] == self.states[next_s][0:2] and a != self.STAY and not is_caught and not exited_maze:
                    rewards[s,a] = self.IMPOSSIBLE_REWARD;
                # Reward for being caught by minotaur
                elif is_caught:
                    rewards[s,a] = self.CAUGHT_REWARD;
                # Reward for reaching the exit
                elif exited_maze:
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
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
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

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;


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

