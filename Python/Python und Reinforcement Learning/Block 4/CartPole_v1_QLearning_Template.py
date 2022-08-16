import numpy as np
import random
import gym
import math
from tqdm import tqdm

MAX_STEPS = 100
MAX_EXP_RATE = 1
MIN_EXP_RATE = 0
EXP_DECAY_RATE = 0.9
TRAIN_EPISODES = 20
LEARNING_RATE = 0.1


class Environment:
    def __init__(self, buckets = (1,1,6,12)):
        # Input:
        #   - self
        #   - buckets: number of discrete intervals for each of the four observations (cart position, cart velocity, pole angle, pole velocity)
        # Return:
        #   - none
        # Function:
        #   - create a new cart pole gym environment
        #   - save the number of buckets for later use
        self.env = gym.make('CartPole-v1')
        # print(self.env.observation_space.low, self.env.observation_space.high)
        self.buckets = buckets

    def discretize(self, obs):
        # Input:
        #   - self
        #   - obs: current state of environment
        # Return:
        #   - new_obs: tuple, in which a new observation is stored
        # Function:
        #   - discretize the continuous observation for cart position, cart velocity, pole angle and pole velocity

        new_obs_0 = np.digitize(obs[0], np.linspace(-2.4, 2.4, self.buckets[0] - 1))
        new_obs_1 = np.digitize(obs[1], np.linspace(-math.inf, math.inf, self.buckets[1] - 1))
        new_obs_2 = np.digitize(obs[2], np.linspace(-15, 15, self.buckets[2] - 1))
        new_obs_3 = np.digitize(obs[3], np.linspace(-30, 30, self.buckets[3] - 1))
        # new_obs_3 = np.digitize(obs[3], np.linspace(-math.inf, math.inf, self.buckets[3] - 1))

        new_obs = [new_obs_0, new_obs_1, new_obs_2, new_obs_3]
        return new_obs


class Agent:
    def __init__(self, lr = LEARNING_RATE, epsilon = 1, episodes = TRAIN_EPISODES):
        # Input:
        #   - self
        #   - lr: Learning rate
        #   - epsilon: epsilon for epsilon-greedy strategy
        #   - episodes: number of episodes
        # Return:
        #   - none
        # Function:
        #   - create variable of class environment
        #   - store all input variables
        #   - create Q-table
        self.Environment = Environment()
        self.lr = lr
        self.epsilon = epsilon
        self.episodes = episodes
        self.gamma = 0.9 # No given gamma value
        self.q_table = dict()  # all Q-values in dictionary
        for x in range(self.Environment.buckets[2]):
            for y in range(self.Environment.buckets[3]):
                self.q_table[(x,y)] = {'LEFT':0, 'RIGHT':0} # Initialize all values with 0

    def update_q_table(self, state, action, reward, new_state):
        # Input:
        #   - self
        #   - state: current state of the environment
        #   - action: chosen action
        #   - reward: reward earned based on the state and action
        #   - new_state: new state of environment based on stae and action
        # Return:
        #   - none
        # Function:
        #   - update value of Q-table for state action tuple
        self.q_table[state][action] = (1 - self.lr) * self.q_table[state][action] + self.lr * (
                    reward + self.gamma * max(self.q_table[new_state].values()))

    def choose_action(self, state, epsilon_greedy = True):
        # Input:
        #   - self
        #   - state: current state of the environment
        #   - epsilon_greedy: bool for choosing strategy (True -> epsilon greedy; False -> best action)
        # Return:
        #   - action: integer which action was chosen by the function
        # Function:
        #   - compute an action based on the input state and the chosen strategy (epsilon-greedy or best action)
        current_state_q = self.q_table[state]
        best_action = []
        actions = ['LEFT', 'RIGHT']
        max_q = max(list(current_state_q.values()))
        for i in range(len(actions)):
            if current_state_q[actions[i]] == max_q:
                best_action.append(actions[i])
        choosed_action = random.choice(best_action)

        if epsilon_greedy:
            R = np.random.random()
            if R > self.epsilon:
                action = choosed_action
            else:
                action = random.choice(actions)
        else:
            action = choosed_action

        return action

    def train(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - train for a maximum of TRAIN_EPISODES
        #   - run each episodes until done==True or the number of MAX_STEPS is reached
        #   - after each episode update epsilon for epsilon-greedy strategy
        #   - after each step update the Q-Table
        for i_episode in range(self.episodes):
            observation = self.Environment.env.reset()
            observation = self.Environment.discretize(observation)
            done = False
            t = 1
            while not done and t <= MAX_STEPS:
                self.Environment.env.render()
                # print(observation)
                state = (observation[2], observation[3])
                action = self.choose_action(state)
                if action == "Left":
                    bi_action = 0
                else:
                    bi_action = 1
                observation, reward, done, info = self.Environment.env.step(bi_action)
                observation = self.Environment.discretize(observation)
                new_state = (observation[2], observation[3])
                self.update_q_table(state, action, reward, new_state)
                t += 1
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                if t == MAX_STEPS:
                    print("Max steps is reached!")
            self.epsilon = (MAX_EXP_RATE - MIN_EXP_RATE) * EXP_DECAY_RATE ** self.episodes

    def play(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - play and render one episode
        #   - print the achieved reward
        sum_reward = 0
        observation = self.Environment.env.reset()
        observation = self.Environment.discretize(observation)
        done = False
        t = 1
        while not done and t <= MAX_STEPS:
           self.Environment.env.render()
           # print(observation)
           state = (observation[2], observation[3])
           action = self.choose_action(state)
           if action == "Left":
               bi_action = 0
           else:
               bi_action = 1
           observation, reward, done, info = self.Environment.env.step(bi_action)
           observation = self.Environment.discretize(observation)
           sum_reward = sum_reward + reward
           t += 1
           if done:
               print("Episode finished after {} timesteps".format(t + 1))
           if t == MAX_STEPS:
               print("Max steps is reached!")
        print("The achieved reward is {}".format(sum_reward))




if __name__ == "__main__":
    agent = Agent()
    agent.train()
    [agent.play() for _ in range(1)]
    print(agent.q_table)