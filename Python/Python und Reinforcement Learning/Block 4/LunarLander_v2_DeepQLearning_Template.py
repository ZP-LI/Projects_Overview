import gym
import keras.models
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
from tf_agents import  replay_buffers

"""
Deep Q-Learning Algorithm for:
    LunarLander_v2
"""

# Define mode (True = No Training):
FLAG_PLAY_ONLY = False

# ToDo: Find suitable values
MAX_TRAINING_EPISODES = 20
MAX_SIZE_BUFFER = 1
MINIBATCH_SIZE = 200
EXP_RATE_MULTIPLIER = 0.9
EXP_RATE_MIN = 0.01
LEARNING_RATE = 0.005
UPDATE_TARGET_AFTER_STEPS = 500
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
            mean_rewards_append = np.average(self.rewards[-100:-1])
        else:
            mean_rewards_append = np.average(self.rewards)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.mean_rewards.append(mean_rewards_append)


    def plot_stats(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Plot reward per episode, mean reward over 100 last episodes and number of steps per episode
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title('Reward per episode')
        ax1.plot(self.rewards)
        ax2.set_title('Mean reward')
        ax2.plot(self.mean_rewards)
        ax3.set_title('Number of steps per episode')
        ax3.plot(self.steps)
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
        #   - Define variables for number of observations and actions, the exploration rate, the discount factor and the learning rate
        #   - Define loss function (Huber loss) and optimizer (Adam)
        #   - Create a replay buffer with maximum size of MAX_SIZE_BUFFER
        #   - Define model and target model for double Deep-Q-Learning
        self.num_obs = num_obs
        self.num_act = num_act
        self.EXP_RATE = 1
        self.gamma = GAMMA
        self.LEARNING_RATE = LEARNING_RATE
        self.loss = tf.keras.losses.Huber()
        self.opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        data_spec = (tf.TensorSpec([8], tf.float32, 'state'), tf.TensorSpec([1], tf.int32, 'action'), tf.TensorSpec([8], tf.float32, 'next_state'), tf.TensorSpec([1], tf.float32, 'reward'), tf.TensorSpec([1], tf.float32, 'done'))
        self.replay_buffer = replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size=MAX_SIZE_BUFFER)
        self.model = keras.models.Sequential(name="Online_Model")
        self.target_model = keras.models.Sequential(name="Target_Model")
        self.model, self.target_model = self.create_net()

    def create_net(self):
        # Input:
        #   - self
        # Return:
        #   - model: Keras model of neural net
        # Function:
        #   - create the neural net: specify the layers and its optimizer and loss function
        self.model.add(keras.Input(shape=self.num_obs))
        self.model.add(keras.layers.Dense(units=128, activation='relu'))
        self.model.add(keras.layers.Dense(units=128, activation='relu'))
        self.model.add(keras.layers.Dense(units=self.num_act, activation='linear'))
        self.model.compile(optimizer=self.opt, loss=self.loss)
        self.model.summary()

        self.target_model.add(keras.Input(shape=self.num_obs))
        self.target_model.add(keras.layers.Dense(units=128, activation='relu'))
        self.target_model.add(keras.layers.Dense(units=128, activation='relu'))
        self.target_model.add(keras.layers.Dense(units=self.num_act, activation='linear'))
        self.target_model.compile(optimizer=self.opt, loss=self.loss)
        self.target_model.summary()

        return self.model, self.target_model

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
        # self.EXP_RATE = (self.EXP_RATE - exp_rate_min) * exp_rate_multiplier ** ep
        self.EXP_RATE = exp_rate_multiplier ** (ep * 1.5) + exp_rate_min

    def eps_greedy(self, state):
        # Input:
        #   - self
        #   - state: Current state of the environment
        # Return:
        #   - [int] action: Chosen action; Integer in range 0 to (num_act - 1)
        # Function:
        #   - performs an epsilon-greedy strategy to choose an action based on the current state and the exploration rate
        state = np.array([state])
        current_state_q = self.model(state)
        current_state_q = current_state_q[0]
        print(current_state_q, self.EXP_RATE)
        best_action = []
        actions = np.arange(self.num_act)
        max_q = max(current_state_q)
        for i in range(len(actions)):
            if current_state_q[i] == max_q:
                best_action.append(actions[i])
        choosed_action = random.choice(best_action)

        R = np.random.random()
        if R > self.EXP_RATE:
            action = choosed_action
        else:
            action = random.choice(actions)

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
        minibatch = self.replay_buffer.get_next(sample_batch_size=MINIBATCH_SIZE, num_steps=1)[0]
        transitions = []
        for i in range(MINIBATCH_SIZE):
            transitions.append((minibatch[0][i][0].numpy(), minibatch[1][i][0][0].numpy(), minibatch[2][i][0].numpy(), minibatch[3][i][0][0].numpy(), minibatch[4][i][0][0].numpy()))

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
        update_q = self.update(states, rewards, actions, dones, next_states, idx)

        if steps >= update_target_after_steps:
            self.model.compile(optimizer=self.opt, loss=self.loss)
            self.target_model.compile(optimizer=self.opt, loss=self.loss)
            self.target_model.fit(states, update_q)
            self.model.fit(states, update_q)

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

        return target_value_ges

    def save_model(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - save the current neural net and its weights to a file
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")

        target_model_json = self.target_model.to_json()
        with open("target_model.json", "w") as json_file:
            json_file.write(target_model_json)
        self.target_model.save_weights("target_model.h5")


    def load_model(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - load a neural net and its weights from a file into the model variable of class Network
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model.h5")

        json_file = open('target_model.json', 'r')
        loaded_target_model_json = json_file.read()
        json_file.close()
        self.target_model = model_from_json(loaded_target_model_json)
        self.target_model.load_weights("target_model.h5")

class TrainSolver:
    def __init__(self, max_episodes=MAX_TRAINING_EPISODES):
        # Input:
        #   - self
        #   - max_episodes: Maximum number of training episodes
        # Return:
        #   - none
        # Function:
        #   - save max_episodes
        #   - create a new environment and get its number of actions and observation
        #   - create a new Network variable
        self.max_episodes = max_episodes
        self.env = gym.make("LunarLander-v2")
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.Network = Network(self.state_size, self.action_size)


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
        step = 0
        steps = 0
        episode = 0
        episodes = 1
        learning_states = LearningStats()
        states = []
        rewards = []
        actions = []
        dones = []
        next_states = []

        while episode < self.max_episodes:
            state = self.env.reset()
            step = 0
            while True:
                self.env.render()
                self.Network.update_exp_rate(episode)
                action = self.Network.eps_greedy(state)
                next_state, reward, done, info = self.env.step(action)
                learning_states.append_step(reward, steps)

                states.append(state)
                rewards.append(reward)
                actions.append(action)
                dones.append(done)
                next_states.append(next_state)

                step += 1
                print(episode, step, action, reward)
                state = next_state
                if done:
                    break

            if rewards[-1] > 10 or rewards[-2] > 10:
                for i in range(len(rewards)):
                    iter_states = tf.convert_to_tensor(states[i], dtype=np.float32)
                    iter_rewards = tf.convert_to_tensor(rewards[i], dtype=np.float32)
                    iter_actions = tf.convert_to_tensor(actions[i], dtype=np.int32)
                    iter_dones = tf.convert_to_tensor(dones[i], dtype=np.float32)
                    iter_next_states = tf.convert_to_tensor(next_states[i], dtype=np.float32)
                    iter_rewards_batched = tf.nest.map_structure(lambda t: tf.expand_dims(t,0), iter_rewards)
                    iter_actions_batched = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), iter_actions)
                    iter_dones_batched = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), iter_dones)
                    values = (iter_states, iter_actions_batched, iter_next_states, iter_rewards_batched, iter_dones_batched)
                    values_batched = tf.nest.map_structure(lambda t: tf.stack([t]), values)
                    self.Network.replay_buffer.add_batch(values_batched)
                steps += step
                episode += 1

            # print(values)
            if steps >= MINIBATCH_SIZE and episodes == episode:
                self.Network.prepare_update(steps)
                self.Network.save_model()
                episodes += 1

        learning_states.plot_stats()

    def play(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Load model from file and play one episode
        #   - Print the achieved score and render the episode
        state = self.env.reset()
        while True:
            self.env.render()
            self.Network.update_exp_rate(1e6)
            action = self.Network.eps_greedy(state)
            next_state, reward, done, info = self.env.step(action)

            state = next_state
            if done:
                break
        print(reward)



if __name__ == "__main__":
    trainer = TrainSolver()

    # don't train, if FLAG_PLAY_ONLY is activated
    if not FLAG_PLAY_ONLY:
        trainer.train()
    trainer.play()
