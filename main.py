import os
import glob
from cart_pole import CartPole
from q_learning_agent import QLearningAgent


def get_episode_count_from_filename(filename):
    return int(os.path.splitext(os.path.basename(filename))[0].split("_")[-1])


def display_available_models():
    """Displays available pre-trained models in the 'models' directory."""
    models_dir = "models"
    files = glob.glob(os.path.join(models_dir, "*.pkl"))
    if not files:
        print("No models found in the 'models' directory.")
        return None

    print("Available Models:")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {os.path.basename(file)}")

    while True:
        try:
            choice = (
                int(input("Select a model to load (enter the corresponding number): "))
                - 1
            )
            model_file = files[choice]
            return model_file
        except (IndexError, ValueError):
            print("Invalid choice. Please enter a number from the list.")


def train_new_model():
    """Trains a new Q-learning agent."""
    gamma = 0.99  # Discount factor
    alpha = 0.1  # Learning rate
    epsilon = 1.0  # Exploration rate
    episodes = int(input("Enter number of episodes to train for: "))

    cart_pole = CartPole(is_learning=True)
    agent = QLearningAgent(
        cart_pole, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes
    )
    agent.apply()
    input("Press any key to continue...")


def continue_training_model(model_file):
    """Continues training a pre-trained Q-learning agent."""
    episode_count = get_episode_count_from_filename(model_file)
    episodes = int(input("Enter number of additional episodes to train for: "))

    cart_pole = CartPole(is_learning=True)
    agent = QLearningAgent(
        cart_pole,
        episodes=episodes,
        is_learning=True,
        model_file=model_file,
        total_episodes_trained=episode_count,
    )
    agent.apply()
    input("Press any key to continue...")


def test_model(model_file):
    """Tests a pre-trained Q-learning agent."""
    episode_count = get_episode_count_from_filename(model_file)
    episodes = int(input("Enter number of episodes to test for: "))

    cart_pole = CartPole(is_learning=False)
    agent = QLearningAgent(
        cart_pole,
        episodes=episodes,
        is_learning=False,
        model_file=model_file,
        total_episodes_trained=episode_count,
    )
    agent.apply()
    input("Press any key to continue...")


def main():
    """Main function for the CartPole Q-learning program."""
    while True:
        print("\nChoose an option:")
        print("1. Train a new model")
        print("2. Continue training a pre-trained model")
        print("3. Test a pre-trained model")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            train_new_model()
        elif choice == "2":
            model_file = display_available_models()
            if model_file:
                continue_training_model(model_file)
        elif choice == "3":
            model_file = display_available_models()
            if model_file:
                test_model(model_file)
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()
