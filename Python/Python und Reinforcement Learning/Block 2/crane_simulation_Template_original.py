import random

import numpy as np
import operator
import matplotlib.pyplot as plt

# Source: Chapter Machine Learning, Lecture ISMLP, AIS 2021

class CraneSim:
    ## Initialize starting data
    def __init__(self):

        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - initialize all required parameters for the crane simulation such as the grid, position of obstacles, ...

        #   - 0 1 2 3 4 5 6 7 8 9
        #   0       
        #   1       
        #   2     O           O
        #   3     O           O
        #   4     O           O
        #   5     O           O
        #   6     O           O
        #   7     O           O
        #   8     O           O
        #   9 S               O T
        # S: Start, O: Obstacle, T: Target
        
        # Size of the problem
        self.height = 10
        self.width = 10
        # self.grid [np-array, dimensions: self.height x self.width] stores the reward the agent earns when going to a specific position
        # start by initializing the reward for all positions with -1
        self.grid = -1 * np.ones((self.height, self.width))

        # define start position: tuple; format: (height, width)
        self.current_location = (9, 0)

        # Obstacles: Save all obstacle positions in a list of tuples
        self.obstacle_list = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
                              (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8)]
        
        # Save the Target Location: tuple
        self.target_location = (9, 9)

        # All terminal states: list of tuples; when reaching those states, the simulation ends
        self.terminal_states = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
                              (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
                                (9, 9)]
        
        # Rewards: set the reward stored in self.grid for all obstacles to -100 and for the target position to +100
        for i in range(len(self.obstacle_list)):
            self.grid[self.obstacle_list[i]] = -100
        self.grid[self.target_location] = +100

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def get_available_actions(self):
        # Returns available actions
        return self.actions
    
    def agent_on_map(self):
        # Returns grid with current position
        grid = np.zeros(( self.height, self.width))
        grid[self.current_location[0],self.current_location[1]] = 1
        return grid
    
    def get_reward(self, new_location):
        # Returns the reward for an input position
        return self.grid[new_location[0], new_location[1]]
    
    def print_current_position(self):
        print("")
        string = "-"
        for y in range(environment.width):
            string += "-" + str(y) + "-"
        print(string) 

        for x in range(environment.height):
            string=str(x)
            for y in range(environment.width):
                ende = False
                for i in range(len(self.obstacle_list)):
                    obstacle = self.obstacle_list[i]
                    if obstacle[0] == x and obstacle[1] == y:
                        string+="-H-"
                        ende = True

                if ende == False:
                    if self.target_location[0] == x and self.target_location[1] == y:
                        string+="-G-"
                    elif self.current_location[0] == x and self.current_location[1] == y:
                        string+="-X-"
                    else:
                        string+="---"

            print(string)
    
    def make_step(self, action):
        # Moves agent; if agent is tries to go over boundary, he doesn't move, but gets a negative reward
        # The function returns the reward for a step

        # Save last location
        last_location = self.current_location
        
        # UP
        if action == 'UP':
            # If agent is on upper boundary, the position doesn't change
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            # Else: Move upwards = increase y-component
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        
        # DOWN
        elif action == 'DOWN':
            # If agent is on lower boundary, the position doesn't change
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
            
        # LEFT
        elif action == 'LEFT':
            # If agent is on left boundary, the position doesn't change
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is on right boundary, the position doesn't change
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)
                
        return reward
    
    def check_state(self):
        # Check if agent is in terminal state (=target or obstacle)
        if self.current_location in self.terminal_states:
            return 'TERMINAL'



class Q_Agent():
    # Initialize
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=0.9):
        self.environment = environment
        self.q_table = dict() # all Q-values in dictionary
        for x in range(environment.height): 
            for y in range(environment.width):
                self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0} # Initialize all values with 0

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def choose_action(self, available_actions,no_random=False):
        # Input:
        #   - self
        #   - available_actions: array of available actions ['UP', 'DOWN', 'LEFT', 'RIGHT']
        #   - no_random: bool
        # Return:
        #   - action: String; has to be one of the available actions out of available_actions
        # Function:
        #   - if no_random is set to true, always return the best action, otherwise use epsilon-greedy strategy as described in slides

        ### Enter your code here ###

        current_state_q = self.q_table[environment.current_location]
        best_action = []
        max_q = max(list(current_state_q.values()))
        for i in range(len(available_actions)):
            if current_state_q[available_actions[i]] == max_q:
                best_action.append(available_actions[i])
        choosed_action = random.choice(best_action)

        if not no_random:
            R = np.random.random()
            if R > self.epsilon:
                action = choosed_action
            else:
                action = random.choice(available_actions)
        else:
            action = choosed_action

        return action

        ### End of your code ###
    
    def update(self, old_state, reward, new_state, action):
        # Input:
        #   - self
        #   - old_state: position in grid before making a step
        #   - reward: earned reward for executed action
        #   - new state: new position in the grid based on old_state and action
        #   - action: chosen action
        # Return:
        #   - none
        # Function:
        #   - Update the Q-Table using the Q-Learning formula

        ### Enter your code here ###

        old_q = self.q_table[old_state][action]

        new_q_vektor = self.q_table[new_state]
        new_state_best_action = max(new_q_vektor)
        new_q = new_q_vektor[new_state_best_action]

        new_choosed_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * new_q)

        self.q_table[old_state][action] = round(new_choosed_q, 3)

        ### End of your code ###



def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, eval = False):
    # Input:
    #   - environment: variable of class CraneSim
    #   - agent: variable of class Q_Agent
    #   - trials: number how many episodes (=games) the Agent will play during the training-process
    #   - max_steps_per_episodes: maximum steps per episode/game/trial
    #   - learn: Bool; shows if Q-Table should be updated
    #   - eval: Bool; shows if Q-Table should only be evaluated
    # Return:
    #   - reward_per_episode: array[number of episodes]; saves the reward earned in each episode
    # Function:
    #   - Run as many trials as specified
    #   - Perform Q-Learning after each action
    #   - If maximum number of steps or a terminal state is reached, save the reward of the current episode, reset the environment and start a new trial
    #   - Distinguish for different states of bool-variables learn and eval
    #   - If eval==True print out the positions of the agent


    # this function iterates and updates the Q-Table, if necessary
    reward_per_episode = [] # Initialize performance log
    
    if eval == True:
        environment.print_current_position()

    for trial in range(trials): # Run trials
        cumulative_reward = 0 # Initialize values of each game
        step = 0
        game_over = False

        ### Enter your code here ###
        while step < max_steps_per_episode:
            if environment.check_state() == 'TERMINAL':
                reward_per_episode.append(cumulative_reward)
                break

            step += 1

            current_action = agent.choose_action(environment.get_available_actions())
            current_state = environment.current_location
            new_reward = environment.make_step(current_action)
            new_state = environment.current_location
            if new_state in environment.obstacle_list:
                game_over = True
            if learn:
                agent.update(current_state, new_reward, new_state, current_action)
            cumulative_reward += new_reward

#            print(current_state, current_action, new_state, new_reward, cumulative_reward, agent.q_table)

        ck = environment.current_location
#        print('-----------------------')


        environment = CraneSim()

        ### End of your code ###

    # Return performance log
    return reward_per_episode, agent.q_table, ck

# Initialize environment and agent
environment = CraneSim()
agentQ = Q_Agent(environment, epsilon=0.05, alpha=0.1, gamma=0.9)

# Train agent
# reward_per_episode, Q_Tables, locations = play(environment, agentQ, trials=10, learn=True, eval = False)
reward_per_episode, Q_Tables, locations = play(environment, agentQ, trials=500, learn=True, eval = False)
# print(Q_Tables)

# Simple learning curve
plt.plot(reward_per_episode)
plt.show()

# Evaluate agent
reward_per_episode, Q_Table, location = play(environment, agentQ, trials=1, learn=False, eval = True)
print(location)