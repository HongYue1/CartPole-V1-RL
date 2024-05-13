import numpy as np
import random
import pickle
import os
import shutil


class QLearningAgent:
    """
    Q-learning agent for CartPole environment.

    Attributes:
        env: The CartPole environment.
        gamma: Discount factor for future rewards.
        alpha: Learning rate.
        epsilon: Exploration rate.
        episodes: Number of episodes to train for.
        is_learning: Boolean indicating whether the agent is in learning mode.
        model_file: Path to a pre-trained model file.
        total_episodes_trained: Total number of episodes the agent has been trained for.
        q_table: The Q-table storing state-action values.
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
                f"Learning mode on: training agent on alpha: {self.alpha}, gamma: {self.gamma}, epsilon : {self.epsilon}, with {self.episodes} episodes"
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
        """Creates the Q-table for the CartPole environment."""
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

        Args:
            state: The current state of the environment.

        Returns:
            The chosen action.
        """
        if self.is_learning and np.random.random() < self.epsilon:
            return random.choice(range(self.env.get_action_space()))
        else:
            return np.argmax(self.q_table[state[0], state[1], state[2], state[3], :])

    def save_checkpoint(self, filename):
        """Saves the current state of the Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Checkpoint saved: {filename}")

    def train(self):
        """Trains the Q-learning agent."""
        os.makedirs("models/checkpoints", exist_ok=True)
        total_episode_rewards = []
        for episode in range(self.episodes):
            episode_rewards = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
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

            self.epsilon -= self.decay_rate
            total_episode_rewards.append(episode_rewards)
            mean_rewards = np.mean(total_episode_rewards[-100:])

            if episode % 100 == 0:
                print(
                    f"Episode: {episode + self.total_episodes_trained} Rewards: {episode_rewards}  Epsilon: {self.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}"
                )

            if mean_rewards >= 1000:
                print(f"Mean rewards: {mean_rewards} - no need to train model longer")
                break

            if episode % 1000 == 0:
                self.save_checkpoint(
                    f"models/checkpoints/Q_table_epoch_{self.total_episodes_trained + episode}.pkl"
                )

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
        """Tests the trained Q-learning agent."""
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

    def apply(self):
        """Starts the training or testing process based on is_learning flag."""
        if self.is_learning:
            self.train()

        else:
            self.test()
