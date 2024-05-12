# CartPole Q-Learning

This project implements a Q-learning agent to solve the classic CartPole problem in the Gymnasium environment. The goal is to train an agent that can balance a pole upright on a moving cart for as long as possible.

## How it Works

The project consists of three main components:

* **`cart_pole.py`**: This file defines the `CartPole` class, which acts as a wrapper for the Gymnasium CartPole environment. It simplifies interaction with the environment and handles the discretization of the continuous state space into bins for use with the Q-table.

* **`q_learning_agent.py`**: This file implements the `QLearningAgent` class, which is the heart of the Q-learning algorithm. The agent learns a Q-table that maps state-action pairs to expected rewards. It uses an epsilon-greedy policy to balance exploration and exploitation during training.

* **`main.py`**: This file provides the main program loop and user interface. It allows the user to train a new model, continue training a pre-trained model, test a model, or exit the program.

## Getting Started

### Prerequisites

* Python 3.6 or later
* Gymnasium
* NumPy

### Installation

1. Clone the repository: `git clone https://github.com/your-username/cartpole-q-learning.git`
2. Install the necessary packages: `pip install gymnasium numpy`

### Usage

1. **Training a new model:**
    * Run `python main.py` and select option 1.
    * Enter the desired number of episodes to train for.
    * The agent will start training and save checkpoints periodically.
    * After training is complete, the final Q-table will be saved in the `models` directory.

2. **Continuing training a pre-trained model:**
    * Run `python main.py` and select option 2.
    * Select the desired model from the list of available models in the `models` directory.
    * Enter the number of additional episodes to train for.
    * The agent will load the Q-table from the selected model and continue training from where it left off.

3. **Testing a pre-trained model:**
    * Run `python main.py` and select option 3.
    * Select the desired model from the list of available models in the `models` directory.
    * Enter the number of episodes to test for.
    * The agent will load the Q-table from the selected model and run for the specified number of episodes, displaying the rewards obtained in each episode.

## Configuration

The Q-learning agent can be configured by modifying the arguments passed to the `QLearningAgent` constructor in `main.py`. The available hyperparameters are:

* **`gamma`**: Discount factor for future rewards (default: 0.99).
* **`alpha`**: Learning rate (default: 0.1).
* **`epsilon`**: Exploration rate (default: 1.0).
* **`episodes`**: Number of episodes to train for.

For example, to create an agent with a learning rate of 0.2 and a discount factor of 0.95, you would modify the code in `main.py` like this:

```python
agent = QLearningAgent(cart_pole, gamma=0.95, alpha=0.2, episodes=episodes)
```

## Results

The trained Q-learning agent is able to balance the pole upright on the cart for a significant number of time steps. The performance of the agent can be evaluated by observing the average reward obtained over a number of episodes during testing.
- optimal table is added in the models folder
