import os
import sys
from atari_env import get_available_atari_games, get_game_config


def display_game_menu():

    available_games = get_available_atari_games()
    
    if not available_games:
        print("No Atari games found. Please check your gym installation.")
        return None
    
    print("\n" + "="*60)
    print("ATARI GAME SELECTOR")
    print("="*60)
    print(f"Found {len(available_games)} available Atari games:")
    print()
    
    # Display games in a grid format
    for i, game in enumerate(available_games, 1):
        print(f"{i:2d}. {game:<20}", end="")
        if i % 3 == 0:
            print()  # New line every 3 games
    
    if len(available_games) % 3 != 0:
        print()  # New line if last row is incomplete
    
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("Enter the number of the game you want to train (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Goodbye!")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_games):
                selected_game = available_games[choice_num - 1]
                print(f"\nSelected: {selected_game}")
                return selected_game
            else:
                print(f"Please enter a number between 1 and {len(available_games)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)


def display_game_info(game_name):

    config = get_game_config(game_name)
    
    print(f"\nGAME INFORMATION: {game_name}")
    print("-" * 40)
    print(f"Fire First: {'Yes' if config['fire_first'] else 'No'}")
    print(f"No-Op Actions: {config['no_ops']}")
    print(f"Action Repeat: {config['repeat']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Epsilon: {config['epsilon']} → {config['min_epsilon']}")
    print(f"Warmup Steps: {config['nb_warmup']}")
    print(f"Memory Capacity: {config['memory_capacity']:,}")
    print(f"Batch Size: {config['batch_size']}")


def get_training_parameters(game_name):

    config = get_game_config(game_name)
    
    print(f"\n TRAINING PARAMETERS")
    print("-" * 40)
    
    # Get number of training epochs
    while True:
        try:
            epochs_input = input(f"Number of training epochs (default: 5000): ").strip()
            if epochs_input == "":
                epochs = 5000
                break
            epochs = int(epochs_input)
            if epochs > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get custom learning rate
    while True:
        try:
            lr_input = input(f"Learning rate (default: {config['learning_rate']}): ").strip()
            if lr_input == "":
                learning_rate = config['learning_rate']
                break
            learning_rate = float(lr_input)
            if learning_rate > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get custom epsilon values
    while True:
        try:
            epsilon_input = input(f"Initial epsilon (default: {config['epsilon']}): ").strip()
            if epsilon_input == "":
                epsilon = config['epsilon']
                break
            epsilon = float(epsilon_input)
            if 0 <= epsilon <= 1:
                break
            else:
                print("Please enter a number between 0 and 1")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            min_epsilon_input = input(f"Minimum epsilon (default: {config['min_epsilon']}): ").strip()
            if min_epsilon_input == "":
                min_epsilon = config['min_epsilon']
                break
            min_epsilon = float(min_epsilon_input)
            if 0 <= min_epsilon <= 1 and min_epsilon <= epsilon:
                break
            else:
                print("Please enter a number between 0 and 1, and less than initial epsilon")
        except ValueError:
            print("Please enter a valid number")
    
    # Get warmup steps
    while True:
        try:
            warmup_input = input(f"Warmup steps (default: {config['nb_warmup']}): ").strip()
            if warmup_input == "":
                nb_warmup = config['nb_warmup']
                break
            nb_warmup = int(warmup_input)
            if nb_warmup >= 0:
                break
            else:
                print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get batch size
    while True:
        try:
            batch_input = input(f"Batch size (default: {config['batch_size']}): ").strip()
            if batch_input == "":
                batch_size = config['batch_size']
                break
            batch_size = int(batch_input)
            if batch_size > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get memory capacity
    while True:
        try:
            memory_input = input(f"Memory capacity (default: {config['memory_capacity']:,}): ").strip()
            if memory_input == "":
                memory_capacity = config['memory_capacity']
                break
            memory_capacity = int(memory_input)
            if memory_capacity > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    return {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'epsilon': epsilon,
        'min_epsilon': min_epsilon,
        'nb_warmup': nb_warmup,
        'batch_size': batch_size,
        'memory_capacity': memory_capacity
    }


def confirm_training(game_name, training_params):

    print(f"\nFINAL TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Game: {game_name}")
    print(f"Training Epochs: {training_params['epochs']:,}")
    print(f"Learning Rate: {training_params['learning_rate']}")
    print(f"Epsilon: {training_params['epsilon']} → {training_params['min_epsilon']}")
    print(f"Warmup Steps: {training_params['nb_warmup']:,}")
    print(f"Batch Size: {training_params['batch_size']}")
    print(f"Memory Capacity: {training_params['memory_capacity']:,}")
    print("=" * 50)
    
    while True:
        confirm = input("\nStart training? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def select_game_and_params():

    # Display available games and get user selection
    game_name = display_game_menu()
    if game_name is None:
        return None, None
    
    # Display game information
    display_game_info(game_name)
    
    # Get training parameters
    training_params = get_training_parameters(game_name)
    
    # Confirm training
    if confirm_training(game_name, training_params):
        return game_name, training_params
    else:
        print("Training cancelled.")
        return None, None


if __name__ == "__main__":
    # Test the game selector
    game_name, params = select_game_and_params()
    if game_name:
        print(f"\nStarting training for {game_name} with parameters: {params}")
    else:
        print("No game selected.") 