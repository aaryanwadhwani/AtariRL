
from atari_env import DQNAtariEnv, get_available_atari_games, get_game_config
from agent import *
from model import *
import torch
import os
import sys

from utils import find_latest_checkpoint, select_model_for_game

def test_model(model_path, game_name="Breakout"):
    
    print(f"Testing model: {model_path}")
    print(f"Game: {game_name}")
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Get game configuration
    game_config = get_game_config(game_name)
    
    try:
        # Create environment
        test_environment = DQNAtariEnv(
            game_name=game_name,
            device=device,
            render_mode='human',
            repeat=game_config['repeat'],
            no_ops=game_config['no_ops'],
            fire_first=game_config['fire_first']
        )
        
        nb_actions = test_environment.get_nb_actions()
        print(f"Number of actions: {nb_actions}")
        
        # Create model with correct number of actions
        model = AtariNet(nb_actions=nb_actions)
        
        # Load the model
        if os.path.exists(model_path):
            # Try to load as new checkpoint format first
            try:
                checkpoint_data = model.load_checkpoint(model_path)
                print(f"Loaded checkpoint: {os.path.basename(model_path)}")
                if 'game_name' in checkpoint_data:
                    print(f"Game: {checkpoint_data['game_name']}")
                if 'epoch' in checkpoint_data:
                    print(f"Epoch: {checkpoint_data['epoch']}")
            except:
                # Fallback to old weight-only format
                model.load_the_model(weights_filename=model_path)
                print(f"Loaded weights: {os.path.basename(model_path)}")
        else:
            print(f"Model file not found: {model_path}")
            return
        
        # Create agent
        agent = Agent(
            model=model,
            device=device,
            epsilon=0.05,
            min_epsilon=0.05,
            nb_warmup=50,
            nb_actions=nb_actions,
            memory_capacity=20000,
            batch_size=32
        )
        
        print(f"Starting test for {game_name}...")
        agent.test(env=test_environment)
        
    except Exception as e:
        print(f"Error testing model: {e}")


def main():
    
    # Default values
    default_game = "Breakout"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        game_name = sys.argv[2] if len(sys.argv) > 2 else default_game
    else:
        # If no model specified, let user choose game and auto-select model
        if len(sys.argv) > 2:
            game_name = sys.argv[2]
        else:
            # If no game specified, let user choose
            available_games = get_available_atari_games()
            if available_games:
                print("Available games for testing:")
                for i, game in enumerate(available_games, 1):
                    print(f"{i:2d}. {game}")
                
                try:
                    choice = input(f"\nSelect game (1-{len(available_games)}) or press Enter for {default_game}: ").strip()
                    if choice:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(available_games):
                            game_name = available_games[choice_num - 1]
                        else:
                            game_name = default_game
                    else:
                        game_name = default_game
                except (ValueError, KeyboardInterrupt):
                    game_name = default_game
            else:
                game_name = default_game
        
        # Automatically find the best model for this game
        model_path = select_model_for_game(game_name)
    
    if model_path:
        print(f"Using model: {os.path.basename(model_path)}")
        print(f"Testing on game: {game_name}")
        # Test the model
        test_model(model_path, game_name)
    else:
        print("No suitable model found for testing")


if __name__ == "__main__":
    main()