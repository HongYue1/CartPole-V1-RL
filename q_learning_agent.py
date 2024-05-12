import numpy as np
import random
import pickle
import os
import shutil


class QLearningAgent:
    """
    Q-learning agent for the CartPole environment.

    This class implements a Q-learning agent that learns a Q-table to determine
    the best action to take in each state of the CartPole environment.

    Attributes:
        env (CartPole): The CartPole environment instance.
        gamma (float): Discount factor for future rewards (default: 0.99).
        alpha (float): Learning rate (default: 0.1).
        epsilon (float): Exploration rate (default: 1.0).
        episodes (int): Number of episodes to train for.
        is_learning (bool): True for training, False for testing.
        decay_rate (float): Rate at which epsilon decays over episodes.
        total_episodes_trained (int): Total episodes the agent has been trained for.
        q_table (np.array): The Q-table storing state-action values.
    """

    def __init__(
        self,
        env,
        gamma=0.99,
        alpha=0.1,
        epsilon=1.0,
        episodes=10000,
        is_learning=True,
        model_file=None,
        total_episodes_trained=0,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.is_learning = is_learning
        self.decay_rate = epsilon / episodes
        self.total_episodes_trained = total_episodes_trained

        if self.is_learning:
            print(
                f"Learning mode on: training agent with alpha: {self.alpha}, "
                f"gamma: {self.gamma}, epsilon: {self.epsilon}, for {self.episodes} episodes"
            )
        else:
            print("Visualization mode on")

        if model_file:
            try:
                with open(model_file, "rb") as f:
                    self.q_table = pickle.load(f)
                print(f"Loaded model: {model_file}")
            except FileNotFoundError:
                print(f"Model file not found: {model_file}")
                exit(1)
        else:
            self.q_table = self._create_q_table()

    def _create_q_table(self):
        """
        Creates the Q-table with dimensions based on the digitized state space.

        The Q-table has dimensions corresponding to the number of bins for each state
        variable (cart position, cart velocity, pole angle, pole angular velocity)
        and the number of possible actions. Each entry in the Q-table represents
        the expected reward for taking a particular action in a particular state.
        """
        pos_space = np.linspace(-2.4, 2.4, 10)
        vel_space = np.linspace(-4, 4, 10)
        ang_space = np.linspace(-0.2095, 0.2095, 10)
        ang_vel_space = np.linspace(-4, 4, 10)
        return np.zeros(
            (
                len(pos_space) + 1,
                len(vel_space) + 1,
                len(ang_space) + 1,
                len(ang_vel_space) + 1,
                self.env.get_action_space(),
            )
        )

    def policy(self, state):
        """
        Chooses an action based on the current state and epsilon-greedy policy.

        With probability epsilon, the agent explores by choosing a random action.
        Otherwise, it exploits by choosing the action with the highest Q-value
        for the given state.

        Args:
            state (list): The current digitized state of the environment.

        Returns:
            int: The chosen action.
        """
        if self.is_learning and np.random.random() < self.epsilon:
            return random.choice(range(self.env.get_action_space()))
        else:
            return np.argmax(self.q_table[state[0], state[1], state[2], state[3], :])

    def save_checkpoint(self, filename):
        """Saves the current Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Checkpoint saved: {filename}")

    def train(self):
        """
        Trains the Q-learning agent using the Q-learning algorithm.

        Runs the agent for a specified number of episodes, updating the Q-table
        based on the rewards received and the agent's actions. Saves checkpoints
        periodically and at the end of training.
        """
        os.makedirs("models/checkpoints", exist_ok=True)
        total_episode_rewards = []

        for episode in range(self.episodes):
            episode_rewards = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _, _ = self.env.step(action)

                # Update Q-table using the Q-learning update rule
                max_next_value = np.max(
                    self.q_table[
                        next_state[0],
                        next_state[1],
                        next_state[2],
                        next_state[3],
                        :,
                    ]
                )
                self.q_table[
                    state[0], state[1], state[2], state[3], action
                ] += self.alpha * (
                    reward
                    + self.gamma * max_next_value
                    - self.q_table[state[0], state[1], state[2], state[3], action]
                )

                episode_rewards += reward
                state = next_state

            self.epsilon -= self.decay_rate  # Decay exploration rate
            total_episode_rewards.append(episode_rewards)
            mean_rewards = np.mean(total_episode_rewards[-100:])

            if episode % 100 == 0:
                print(
                    f"Episode: {episode + self.total_episodes_trained} "
                    f"Rewards: {episode_rewards}  Epsilon: {self.epsilon:0.2f}  "
                    f"Mean Rewards {mean_rewards:0.1f}"
                )

            if mean_rewards >= 1000:
                print(f"Mean rewards: {mean_rewards} - no need to train model longer")
                break

            # Save checkpoint every 1000 episodes
            if episode % 1000 == 0:
                self.save_checkpoint(
                    f"models/checkpoints/Q_table_epoch_{self.total_episodes_trained + episode}.pkl"
                )

        # Save final model
        self.save_checkpoint(
            f"models/Q_table_{self.total_episodes_trained + self.episodes}.pkl"
        )
        print(
            f"Final model saved: models/Q_table_{self.total_episodes_trained + self.episodes}.pkl"
        )

        # Cleanup: Delete all checkpoint files except the final model
        shutil.rmtree("models/checkpoints")
        print("All checkpoint files deleted.")

    def test(self):
        """
        Tests the trained Q-learning agent.

        Runs the agent in the environment for a specified number of episodes,
        using the learned Q-table to choose actions, and prints the episode rewards.
        """
        for episode in range(self.episodes):
            episode_rewards = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_rewards += reward
                state = next_state
            print(f"Episode: {episode} Rewards: {episode_rewards}")

        self.env.env.close()

    def apply(self):
        """Starts the training or testing process based on is_learning flag."""
        if self.is_learning:
            self.train()
        else:
            self.test()
