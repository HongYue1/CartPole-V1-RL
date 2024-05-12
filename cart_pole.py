import gymnasium as gym
import numpy as np


class CartPole:
    """
    Wrapper class for the CartPole environment.

    This class provides a simplified interface for interacting with the CartPole
    environment from Gymnasium, including state digitization for use with Q-learning.

    Attributes:
        env: The Gymnasium environment for the Cart Pole game.
        curr_state (np.array): The current state of the environment, digitized.
        is_terminated (bool): Flag indicating whether the current episode has ended.
    """

    def __init__(self, is_learning=False):
        """
        Initializes the CartPole environment.

        Args:
            is_learning (bool): Flag to determine if the environment is for learning
                              or visualization. If True, no rendering is performed.
        """
        if is_learning:
            self.env = gym.make("CartPole-v1")
        else:
            self.env = gym.make("CartPole-v1", render_mode="human")

        self.curr_state = self.digitize_state(self.env.reset()[0])
        self.is_terminated = False

    def digitize_state(self, state):
        """
        Digitizes the continuous state variables into discrete values for Q-table.

        The state variables (cart position, cart velocity, pole angle, pole angular
        velocity) are each divided into 10 bins.

        Args:
            state (np.array): The current state of the environment as a 1D array.

        Returns:
            list: A list representing the digitized state, where each element
                  corresponds to the bin index for each state variable.
        """
        pos_space = np.linspace(-2.4, 2.4, 10)
        vel_space = np.linspace(-4, 4, 10)
        ang_space = np.linspace(-0.2095, 0.2095, 10)
        ang_vel_space = np.linspace(-4, 4, 10)

        new_state_p = np.digitize(state[0], pos_space)
        new_state_v = np.digitize(state[1], vel_space)
        new_state_a = np.digitize(state[2], ang_space)
        new_state_av = np.digitize(state[3], ang_vel_space)

        return [new_state_p, new_state_v, new_state_a, new_state_av]

    def step(self, action):
        """
        Performs a step in the environment by taking the given action.

        This method updates the environment based on the action, digitizes
        the resulting state, and returns the new state, reward, termination
        status, and additional info.

        Args:
            action (int): An action from the action space (0 or 1)
                         representing moving left or right.

        Returns:
            tuple: A tuple containing:
                - new_state (list): The digitized state after taking the action.
                - reward (float): The reward received for taking the action.
                - done (bool): True if the episode is terminated, False otherwise.
                - info (dict): Additional information from the environment.
        """
        new_state, reward, self.is_terminated, _, _ = self.env.step(action)
        self.curr_state = self.digitize_state(new_state)
        return self.curr_state, reward, self.is_terminated, _, _

    def reset(self):
        """Resets the environment to its initial state."""
        self.curr_state = self.digitize_state(self.env.reset()[0])
        self.is_terminated = False
        return self.curr_state

    def get_action_space(self):
        """Returns the size of the action space (number of possible actions)."""
        return self.env.action_space.n
