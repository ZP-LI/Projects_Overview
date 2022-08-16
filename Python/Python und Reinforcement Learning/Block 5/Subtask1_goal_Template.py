# Source: https://gitlab.lrz.de/TUMWAIS/public/xppusim

import numpy as np
from rllib.classes.goals.xppu.GoalAbstract import GoalAbstract

MAX_STEPS = 100


class Subtask1_goal(GoalAbstract):
    """
    Scenario to move a single workpiece from the stack to the beginning of the large sorting conveyor.
    White and metallic workpieces have to be stamped, black workpieces have to remain unstamped.
    The task is considered finished, when the workpiece is put down on the conveyor.
    If the stamping matches the workpiece color, the goal was achieved.

    Additional information:
    desired_goal and achieved_goal always have to be arrays consisting of at least 3 integers:
    e.g. np.array([1, 1, 1]) or np.array([0, 0, 0])


    """

    def __init__(self,
                 state_space_mapping,
                 info,
                 scaling_factor=1,
                 error_penalty=1):
        """
        Initializes the goal object

        :param state_space_mapping: Dictionary containing the state space mapping of the environment
        :param info:                Info dict of the environment after initialization
        :param scaling_factor:      Integer for scaling the reward. If not provided, the max. reward is 1
        :param error_penalty:       Integer representing the penalty for errors
        """
        super().__init__(state_space_mapping, info)

        self.scaling_factor = scaling_factor
        self.error_penalty = error_penalty

    def reset(self, observation, info):
        """
        Resets the goal class before a new trajectory is started.
        In particular, the desired goal and the maximum possible reward is initialized.

        :param observation:     Observation vector for the current simulation state
        :param info:            Info dictionary for the current simulation state
        :return achieved_goal:  Numpy array with the achieved goal
        :return desired_goal:   Numpy array representing the desired goal
        """
        super().reset(observation, info)

        self.steps = 0

        # Update desired goal and maximum possible reward
        self.desired_goal = np.array([1, 1, 1])
        self.max_reward = 100

        self.max_reward *= self.scaling_factor

        self.achieved_goal, self.desired_goal = self.update(observation, info)

        return self.achieved_goal, self.desired_goal

    def update(self, observation, info):
        """
        Updates the achieved goal after each transition

        :param observation:     Observation vector for the current simulation state
        :param info:            Info dictionary for the current simulation state
        :return achieved_goal:  Numpy array with the achieved goal
        :return desired_goal:   Numpy array representing the desired goal
        """
        super().update(observation, info)

        # ToDo: return achieved_goal = np.array([1, 1, 1]), if goal was achieved. Otherwise return achieved_goal = np.array([0, 0, 0])

        return self.achieved_goal, self.desired_goal

    def compute_reward(self, prev_achieved_goal, achieved_goal, desired_goal, info):
        """
        Returns the reward for the transition defined by prev_achieved_goal, achieved_goal, and desired_goal

        :param prev_achieved_goal:  Numpy array of achieved goal before transition
        :param achieved_goal:       Numpy array of achieved goal after transition
        :param desired_goal:        Numpy array of desired goal
        :param info:                Info dictionary after transition
        :return reward:             Integer represention reward for the transition
        :return done:               Boolean value indicating whether the trajectory is finished
        :return info:               Updated info dict (reward_msg, done_msg, error_msg and max_reward)
        """
        # Each step costs 0.1 point
        reward = -0.1
        # Update step counter
        self.steps += 1

        done = False
        info = info.copy()  # Don't change original dict (necessary for HER)
        info["max_reward"] = self.max_reward

        # ToDo: Compute reward for executed step. Set done to true, if task is finished or maximum number of steps is reached
        # Workpiece reached conveyor and was stamped the right way: +100 reward
        # Workpiece reached conveyor and was NOT stamped the right way: -100 reward


    def _get_goal_space_mapping(self, info):
        """
        Returns a list that maps the indices of the goal vector into workpiece-specific goal groups as well as a
        tuple describing the shape of the goal.

        [Workpiece index]
            [Goal variable]: [Index in the goal vector]

        :return goal_space_mapping: List of dictionaries containing the mapping
        :return shape: Tuple containing the shape of the goal
        """

        goal_space_mapping, shape = super()._get_goal_space_mapping(info)

        VARS_PER_WP = 1
        num_wps = len(info["workpiece_info"])

        for wp_idx in range(num_wps):
            offset = VARS_PER_WP * wp_idx
            goal_space_mapping["workpiece"].append({"stored": 0 + offset})

        shape = (VARS_PER_WP * num_wps,)

        return goal_space_mapping, shape

