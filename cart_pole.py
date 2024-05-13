import gymnasium as gym
import numpy as np


class CartPole:
    """
    Wrapper class for CartPole environment.

    Attributes:
        env: The Gym environment for the Cart Pole game.
        curr_state (np.array): The current state of the environment.
        is_terminated (bool): Flag indicating whether the current episode has ended.
    """

    def __init__(self, is_learning=False):
        """
        Initializes the CartPole environment.

        Args:
            is_learning (bool): Flag to determine if the environment is for learning or visualization.
        """
        # Define whether we want to visualize
        if is_learning:
            self.env = gym.make("CartPole-v1")
        else:
            self.env = gym.make("CartPole-v1", render_mode="human")
        self.curr_state = self.digitize_state(self.env.reset()[0])
        self.is_terminated = False

    def digitize_state(self, state):
        """
        Digitizes the continuous state into discrete values for Q-table.

        Args:
            state (np.array): The current state of the environment.

        Returns:
            list: A list representing the digitized state.
        """
        pos_space = np.linspace(-2.4, 2.4, 10)
        vel_space = np.linspace(-4, 4, 10)
        ang_space = np.linspace(-0.2095, 0.2095, 10)
        ang_vel_space = np.linspace(-4, 4, 10)

        new_state_p = np.digitize(state[0], pos_space)
        new_state_v = np.digitize(state[1], vel_space)
        new_state_a = np.digitize(state[2], ang_space)
        new_state_av = np.digitize(state[3], ang_vel_space)
        new_state_dig = [new_state_p, new_state_v, new_state_a, new_state_av]
        return new_state_dig

    def step(self, action):
        """
        Performs a step in the environment. Gets the values for observation, reward,
        and checks if the game is over.

        Args:
            action (int): An action passed to the environment.

        Returns:
            new_state: Discrete state after the action is taken.
            reward: Reward based on the taken action.
            done: Boolean indicating whether the episode is terminated.
            info: Additional information from the environment.
        """
        new_state, reward, self.is_terminated, _, _ = self.env.step(action)
        # Update the current state
        self.curr_state = self.digitize_state(new_state)
        return self.curr_state, reward, self.is_terminated, _, _

    def reset(self):
        """Resets the environment."""
        self.curr_state = self.digitize_state(self.env.reset()[0])
        self.is_terminated = False
        return self.curr_state

    def get_action_space(self):
        """Returns the size of the action space."""
        return self.env.action_space.n
