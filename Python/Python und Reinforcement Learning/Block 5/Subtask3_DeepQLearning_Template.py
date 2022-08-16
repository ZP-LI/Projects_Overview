# Source: https://gitlab.lrz.de/TUMWAIS/public/xppusim

import os
from xppusim.gym import gym_env
from Subtask3_goal import Subtask3_goal
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

"""
Deep Q-Learning Algorithm for:
    xPPUSim Subtask3
"""

# Define mode (True = No Training):
FLAG_PLAY_ONLY = False

# Choose Goal
GOAL_SELECT = Subtask3_goal
SUCCESS = 99

# Specify paths and files for statespace and actionspace
STATE_SPACE_PATH = 'Subtask3_statespace.JSON'
ACTION_SPACE_PATH = 'Subtask3_actionspace.JSON'

# ToDo: Find suitable values
MAX_TRAINING_EPISODES =
MAX_SIZE_BUFFER =
MINIBATCH_SIZE =
EXP_RATE_MULTIPLIER =
EXP_RATE_MIN =
LEARNING_RATE =
UPDATE_TARGET_AFTER_STEPS =
GAMMA =


class LearningStats:
    def __init__(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function: define empty list for rewards, mean_rewards and steps


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


    def plot_stats(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Plot reward per episode, mean reward over 100 last episodes and number of steps per episode



class Network:
    def __init__(self, num_obs, num_act):
        # Input:
        #   - self
        #   - num_obs: Number of observations of environment
        #   - num_act: Number of actions available in the environment
        # Return:
        #   - none
        # Function:
        #   - Define variables for number of observations and actions, the exploration rate, the discount factor and the learning rate
        #   - Define loss function (Huber loss) and optimizer (Adam)
        #   - Create a replay buffer with maximum size of MAX_SIZE_BUFFER
        #   - Define model and target model for double Deep-Q-Learning

    def create_net(self):
        # Input:
        #   - self
        # Return:
        #   - model: Keras model of neural net
        # Function:
        #   - create the neural net: specify the layers and its optimizer and loss function


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

    def eps_greedy(self, state):
        # Input:
        #   - self
        #   - state: Current state of the environment
        # Return:
        #   - [int] action: Chosen action; Integer in range 0 to (num_act - 1)
        # Function:
        #   - performs an epsilon-greedy strategy to choose an action based on the current state and the exploration rate


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
        transitions =


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

    def load_model(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - load a neural net and its weights from a file into the model variable of class Network


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
                                      wp_combination_pool_size=3,
                                      wp_num_in_stack=1,
                                      wp_pool_seed=None,
                                      render=render_bool,
                                      normalize_state_space=False,
                                      action_space_type="high_level",
                                      position_list=None,
                                      render_dir=os.getcwd() + '/output',
                                      crane_turn_only_extended=True,
                                      crane_move_at_pickpoint=True,
                                      reset_fn=None)


    def train(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - initialize step counter and learning stats
        #   - Train the net for a maximum of self.max_episodes
        #   - Take care of/remember:
        #       - there is no training if the number of samples in the replay buffer is lower than the desired number of transitions in each minibatch
        #       - exploration rate
        #       - the order of the variables that are written to the replay buffer
        #       - at the end of each episode save the results to stats
        #       - you can terminate training, if the goal/success is reached
        #       - save the model after training
        #       - plot the training stats
        #       - Remember that Subtask1 should be executed by the algorithm of Subtask3 in order to complete the whole task.
        #       - You may use a hardcoded solution for Subtask1, but it has to be executed by the RL-algorithm



    def play(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Load model from file and play one episode
        #   - Print the achieved score and render the episode



if __name__ == "__main__":
    # don't train, if FLAG_PLAY_ONLY is activated
    if not FLAG_PLAY_ONLY:
        trainer = TrainSolver()
        trainer.train()
    trainer = TrainSolver(render_bool=True)
    trainer.play()
