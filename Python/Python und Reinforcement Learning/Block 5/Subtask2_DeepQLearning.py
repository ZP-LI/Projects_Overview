# Source: https://gitlab.lrz.de/TUMWAIS/public/xppusim

import os
import sys

from xppusim.gym import gym_env
from Subtask2_goal import Subtask2_goal
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

"""
Deep Q-Learning Algorithm for:
    xPPUSim Subtask2
"""

# Define mode (True = No Training):
FLAG_PLAY_ONLY = False

# Choose Goal
GOAL_SELECT = Subtask2_goal
SUCCESS = 150  # 99

# Specify paths and files for statespace and actionspace
STATE_SPACE_PATH = 'Subtask2_statespace.JSON'
ACTION_SPACE_PATH = 'Subtask2_actionspace.JSON'

# ToDo: Find suitable values
MAX_TRAINING_EPISODES = 1000
MAX_SIZE_BUFFER = 1000000
MINIBATCH_SIZE = 64
EXP_RATE_MULTIPLIER = 0.996
EXP_RATE_MIN = 0.1
LEARNING_RATE = 0.001
UPDATE_TARGET_AFTER_STEPS = 300
GAMMA = 0.99


class LearningStats:
    def __init__(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function: define empty list for rewards, mean_rewards and steps
        self.rewards = []
        self.mean_rewards = []
        self.steps = []

    def append_step(self, reward, steps):
        # Input:
        #   - self
        #   - reward: reward of episode
        #   - steps: number of steps of episode
        # Return:
        #   - none
        # Function:
        #   - Compute mean reward for last 100 episodes
        #   - Append reward, steps and mean reward to corresponding list
        if len(self.rewards) > 100:
            mean_rewards = np.average(self.rewards[-100:])
        else:
            mean_rewards = np.average(self.rewards)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.mean_rewards.append(mean_rewards)

    def plot_stats(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Plot reward per episode, mean reward over 100 last episodes and number of steps per episode
        plt.subplot(3, 1, 1)
        plt.title('Reward per episode')
        plt.plot(self.rewards)

        plt.subplot(3, 1, 2)
        plt.title('Mean reward')
        plt.plot(self.mean_rewards)

        plt.subplot(3, 1, 3)
        plt.title('Number of steps per episode')
        plt.plot(self.steps)

        plt.show()


class Network:
    def __init__(self, num_obs, num_act):
        # Input:
        #   - self
        #   - num_obs: Number of observations of environment
        #   - num_act: Number of actions available in the environment
        # Return:
        #   - none
        # Function:
        #   - Define variables for number of observations and actions, the exploration rate,
        #     the discount factor and the learning rate
        #   - Define loss function (Huber loss) and optimizer (Adam)
        #   - Create a replay buffer with maximum size of MAX_SIZE_BUFFER
        #   - Define model and target model for double Deep-Q-Learning
        self.num_obs = num_obs
        self.num_act = num_act
        self.exp_rate = 1.0
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        # self.loss = tf.keras.losses.Huber()
        self.loss = tf.keras.losses.Huber()
        self.opt = Adam(learning_rate=self.learning_rate)
        self.replay_buffer = deque(maxlen=MAX_SIZE_BUFFER)
        self.model = self.create_net()
        self.target_model = self.create_net()

    def create_net(self):
        # Input:
        #   - self
        # Return:
        #   - model: Keras model of neural net
        # Function:
        #   - create the neural net: specify the layers and its optimizer and loss function
        model = Sequential()
        model.add(Dense(units=128, input_dim=self.num_obs, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.num_act, activation='linear'))
        model.compile(optimizer=self.opt, loss=self.loss)
        model.summary()

        return model

    def update_exp_rate(self, ep, exp_rate_multiplier=EXP_RATE_MULTIPLIER, exp_rate_min=EXP_RATE_MIN):
        # Input:
        #   - self
        #   - ep: Number of current episode
        #   - exp_rate_multiplier: Exploration rate multiplier
        #   - exp_rate_min: Minimum value for exploration rate
        # Return:
        #   - none
        # Function:
        #   - Compute and update the exploration rate
        self.exp_rate = max(exp_rate_min, exp_rate_multiplier * self.exp_rate)

    def eps_greedy(self, state):
        # Input:
        #   - self
        #   - state: Current state of the environment
        # Return:
        #   - [int] action: Chosen action; Integer in range 0 to (num_act - 1)
        # Function:
        #   - performs an epsilon-greedy strategy to choose an action based on the current state
        #     and the exploration rate
        q_hat = self.model(np.array([state]))[0]
        # get all possible best actions
        max_q = max(q_hat)
        best_actions = []
        for i in range(self.num_act):
            if q_hat[i] == max_q:
                best_actions.append(i)

        if np.random.uniform(0.0, 1.0) < self.exp_rate:
            action = np.random.choice(self.num_act)
        else:
            action = random.choice(best_actions)

        return action

    def prepare_update(self, steps, update_target_after_steps=UPDATE_TARGET_AFTER_STEPS):
        # Input:
        #   - self
        #   - steps: total number of steps in training process so far
        #   - update_target_after_steps: Number of steps after which target model is updated
        # Return:
        #   - none
        # Function:
        #   - Sample a minibatch from replay buffer and save it in transitions
        #   - Update target network if necessary
        cur_batch_size = min(len(self.replay_buffer), MINIBATCH_SIZE)
        transitions = random.sample(self.replay_buffer, cur_batch_size)

        # DO NOT TOUCH!
        idx = np.array([i for i in range(MINIBATCH_SIZE)])
        states = [i[0] for i in transitions]
        actions = [i[1] for i in transitions]
        next_states = [i[2] for i in transitions]
        rewards = [i[3] for i in transitions]
        dones = [i[4] for i in transitions]

        # Convert inputs from mini_batch to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)

        # separate function which is defined as tf.function to improve computing speed
        self.update(states, rewards, actions, dones, next_states, idx)

        # update target model
        if steps % update_target_after_steps == 0:
            self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def update(self, states, rewards, actions, dones, next_states, idx):
        # compute forward pass for next states
        target_q_sel = self.model(next_states)
        # get next action
        next_action = tf.argmax(target_q_sel, axis=1)

        # apply double Deep-Q-Learning
        # compute forward pass for next states on target network
        target_q = self.target_model(next_states)
        # create array [#states x #actions]: for selected action write target_q value, for other actions write 0
        target_value = tf.reduce_sum(tf.one_hot(next_action, self.num_act) * target_q, axis=1)

        # Q-values of successor states plus reward for current transition
        # if done=1 -> no successor -> consider current reward only
        target_value_update = (1 - dones) * self.gamma * target_value + rewards
        target_value_orig = self.model(states)

        # update target_value_orig with target_value_update on positions of chosen actions
        target_value_ges = tf.tensor_scatter_nd_update(target_value_orig, tf.stack([idx, actions], axis=1),
                                                       target_value_update)

        # get all trainable variables of the model
        dqn_variable = self.model.trainable_variables

        with tf.GradientTape() as tape:
            # forward pass for states
            logits = self.model(states)
            # compute loss
            loss = self.loss(tf.stop_gradient(target_value_ges), logits)

        # compute gradient
        dqn_grads = tape.gradient(loss, dqn_variable)

        # apply gradient
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

    def save_model(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - save the current neural net and its weights to a file
        self.model.save("model")
        self.target_model.save("target_model")

    def load_model(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - load a neural net and its weights from a file into the model variable of class Network
        self.model = tf.keras.models.load_model("model")


class TrainSolver:
    def __init__(self, render_bool=False, max_episodes=MAX_TRAINING_EPISODES, success=SUCCESS):
        # Input:
        #   - self
        #   - render_bool: only render simulation, if true (real slow, so only use for play mode)
        #   - max_episodes: Maximum number of training episodes
        #   - success: Mean reward over last 100 episodes which is needed to consider task as solved
        # Return:
        #   - none
        # Function:
        #   - save max_episodes
        #   - create a new environment (already done) and get its number of actions and observation
        #   - create a new Network variable

        self.env = gym_env.XPPUSIMEnv(state_space_conf_path=STATE_SPACE_PATH,
                                      action_space_conf_path=ACTION_SPACE_PATH,
                                      goal=GOAL_SELECT,
                                      goal_args=[],
                                      wp_in_stack=None,
                                      wp_combinations_in_pool=None,
                                      wp_combination_pool_size=27,
                                      wp_num_in_stack=3,
                                      wp_pool_seed=None,
                                      render=render_bool,
                                      normalize_state_space=False,
                                      action_space_type="high_level",
                                      position_list=None,
                                      render_dir=os.getcwd() + '/output',
                                      crane_turn_only_extended=True,
                                      crane_move_at_pickpoint=True,
                                      reset_fn=None)
        self.max_episodes = max_episodes
        self.action_size = self.env.action_space.n - 5
        self.state_size = self.env.get_observation()[0]['observation'].shape[0]
        # print(self.env.get_observation())
        self.network = Network(self.state_size, self.action_size)
        self.success_mean_reward = success

    def train(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - initialize step counter and learning stats
        #   - Train the net for a maximum of self.max_episodes
        #   - Take care of/remember:
        #       - there is no training if the number of samples in the replay buffer is
        #         lower than the desired number of transitions in each minibatch
        #       - exploration rate
        #       - the order of the variables that are written to the replay buffer
        #       - at the end of each episode save the results to stats
        #       - you can terminate training, if the goal/success is reached
        #       - save the model after training
        #       - plot the training stats
        #       - Remember that Subtask1 should be executed by the algorithm of Subtask2 in order to
        #         complete the whole task.
        #       - You may use a hardcoded solution for Subtask1, but it has to be executed by the RL-algorithm
        #       - Check whether chosen action is suitable for current workpiece position and only
        #         execute action if it's suitable
        step = 0
        learning_stats = LearningStats()
        timer = time.perf_counter()
        episode = 0

        ramp1_pos = [74.40, 71.51, 6.79]
        ramp2_pos = [74.40, 99.24, 6.79]
        lsc_pos0 = [74.40, 48.95, 6.79]

        while episode < self.max_episodes:
            self.env.reset()
            # state = self.env.get_observation()[0]['observation']
            episode_reward = 0
            episode_step = 0
            clock = 0
            start_time = time.perf_counter()
            done = False
            stored = 0

            wp_num = 0

            while not done:
                # self.env.render()
                self.env.step([0])
                self.env.step([1])
                self.env.step([3])
                self.env.step([2])
                self.env.step([4])
                state = self.env.get_observation()[0]['observation']
                prev_stored = stored

                while stored == prev_stored and not done:
                    # print("ep {} step {} state: {}".format(episode, episode_step, state))
                    action = 0

                    if ((lsc_pos0[0] - 0.1) < state[wp_num * 6 + 0] < (lsc_pos0[0] + 0.1)) and (
                            (lsc_pos0[1] - 0.1) < state[wp_num * 6 + 1] < (lsc_pos0[1] + 0.1)) and (
                            (lsc_pos0[2] - 0.1) < state[wp_num * 6 + 2] < (lsc_pos0[2] + 0.1)):
                        action = 0
                    elif action == 3 and ((ramp1_pos[0] - 0.1) < state[wp_num * 6 + 0] < (ramp1_pos[0] + 0.1)) and (
                            (ramp1_pos[1] - 0.1) < state[wp_num * 6 + 1] < (ramp1_pos[1] + 0.1)) and (
                            (ramp1_pos[2] - 0.1) < state[wp_num * 6 + 2] < (ramp1_pos[2] + 0.1)):
                        action = 5
                    elif action == 4 and ((ramp2_pos[0] - 0.1) < state[wp_num * 6 + 0] < (ramp2_pos[0] + 0.1)) and (
                            (ramp2_pos[1] - 0.1) < state[wp_num * 6 + 1] < (ramp2_pos[1] + 0.1)) and (
                            (ramp2_pos[2] - 0.1) < state[wp_num * 6 + 2] < (ramp2_pos[2] + 0.1)):
                        action = 6
                    else:
                        prev_action = action
                        while prev_action == action:
                            action = self.network.eps_greedy(state)

                    action = self.network.eps_greedy(state)
                    next_state, reward, done, info = self.env.step([action + 5])
                    next_state = next_state['observation']

                    if step > MINIBATCH_SIZE:
                        self.network.prepare_update(step)
                    self.network.replay_buffer.append((state, action, next_state, reward, done))
                    episode_step += 1
                    step += 1
                    clock = time.perf_counter() - start_time
                    episode_reward += reward
                    state = next_state
                    stored = np.sum(self.env.get_observation()[0]['achieved_goal'])
                    # print('prev_stored {} / stored {}'.format(prev_stored, stored))

                wp_num += 1

            print("episode {} exploration rate {:.4f} reward: {:.2f} step {} time: {:.2f}s".
                  format(episode, self.network.exp_rate, episode_reward, episode_step, clock))
            print(self.env.get_observation()[0]['achieved_goal'])

            episode += 1
            self.network.update_exp_rate(episode)
            learning_stats.append_step(episode_reward, episode_step)
            self.network.save_model()
            if learning_stats.mean_rewards[-1] > self.success_mean_reward:
                break

        learning_stats.plot_stats()
        print("Total training time {:.2f}s".format(time.perf_counter() - timer))

    def play(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Load model from file and play one episode
        #   - Print the achieved score and render the episode
        self.network.load_model()
        self.network.exp_rate = 0.1
        state = self.env.reset()[0]['observation']

        step = 0
        total_reward = 0
        done = False
        stored = 0

        print("Start playing")
        while not done:
            # self.env.render()
            self.env.step([0])
            self.env.step([1])
            self.env.step([3])
            self.env.step([2])
            self.env.step([4])
            state = self.env.get_observation()[0]['observation']
            prev_stored = stored

            while stored == prev_stored and not done:
                action = self.network.eps_greedy(state)
                print("{}. Step, current total reward {:.2f} action {}\n achieved goal {}".
                      format(step, total_reward, action, self.env.get_observation()[0]['achieved_goal']))

                next_state, reward, done, info = self.env.step([action + 5])
                next_state = next_state['observation']
                state = next_state
                stored = np.sum(self.env.get_observation()[0]['achieved_goal'])

                total_reward += reward
                step += 1

        print("Achieved goal {}\nPlay reward {:.3f}".format(self.env.get_observation()[0]['achieved_goal'],
                                                            total_reward))


if __name__ == "__main__":
    # don't train, if FLAG_PLAY_ONLY is activated
    if not FLAG_PLAY_ONLY:
        trainer = TrainSolver()
        trainer.train()
    trainer = TrainSolver(render_bool=True)
    trainer.play()
