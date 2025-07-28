# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from atari_env import DQNAtariEnv, get_game_config
from agent import *
from model import *
from game_selector import select_game_and_params
import torch
import os
import glob
import re

from utils import find_latest_checkpoint, list_checkpoints

def main():
    
    # Set environment variable for compatibility
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get game selection and parameters from user
    game_name, training_params = select_game_and_params()
    
    if game_name is None:
        print("No game selected. Exiting.")
        return
    
    print(f"\nStarting training for {game_name}...")
    
    # Get game-specific configuration
    game_config = get_game_config(game_name)
    
    # Create environment
    try:
        environment = DQNAtariEnv(
            game_name=game_name,
            device=device,
            repeat=game_config['repeat'],
            no_ops=game_config['no_ops'],
            fire_first=game_config['fire_first']
        )
        print(f"Environment created successfully for {game_name}")
        print(f"Number of actions: {environment.get_nb_actions()}")
    except Exception as e:
        print(f"Failed to create environment for {game_name}: {e}")
        return
    
    # Create model with correct number of actions
    nb_actions = environment.get_nb_actions()
    model = AtariNet(nb_actions=nb_actions)
    
    # Resume logic
    resume = False
    start_epoch = 1
    stats = None
    optimizer_state_dict = None
    epsilon = training_params['epsilon']
    
    # Check for existing checkpoints
    checkpoints = list_checkpoints(game_name)
    if checkpoints:
        print(f"\nFound {len(checkpoints)} checkpoints for {game_name}.")
        print("Most recent:")
        for i, ckpt in enumerate(checkpoints[:5], 1):
            print(f"  {i}. {os.path.basename(ckpt)}")
        print("  ..." if len(checkpoints) > 5 else "")
        choice = input("\nResume from latest checkpoint? (y/n) Or enter number to pick: ").strip().lower()
        if choice in ['y', 'yes', '']:
            checkpoint_path = checkpoints[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
            checkpoint_path = checkpoints[int(choice)-1]
        else:
            checkpoint_path = None
        if checkpoint_path:
            print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
            # Load checkpoint
            out = model.load_checkpoint(checkpoint_path)
            if 'optimizer_state_dict' in out:
                optimizer_state_dict = out['optimizer_state_dict']
            if 'epoch' in out:
                start_epoch = out['epoch'] + 1
            if 'epsilon' in out:
                epsilon = out['epsilon']
            if 'stats' in out:
                stats = out['stats']
            resume = True
            print(f"Resuming from epoch {start_epoch}")
    
    # Create agent with user-specified or loaded parameters
    agent = Agent(
        model=model,
        device=device,
        epsilon=epsilon,
        min_epsilon=training_params['min_epsilon'],
        nb_warmup=training_params['nb_warmup'],
        nb_actions=nb_actions,
        learning_rate=training_params['learning_rate'],
        memory_capacity=training_params['memory_capacity'],
        batch_size=training_params['batch_size'],
        optimizer_state_dict=optimizer_state_dict
    )
    
    print(f"\nStarting training for {training_params['epochs']} epochs...")
    print("Training progress will be logged to TensorBoard")
    
    # Train the agent
    try:
        stats = agent.train(
            env=environment,
            epochs=training_params['epochs'],
            batch_identifier=game_name,
            start_epoch=start_epoch,
            stats=stats,
            game_name=game_name,
            game_config=game_config,
            resume_checkpoint_dir="models"
        )
        print(f"\nTraining completed for {game_name}!")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted for {game_name}")
    except Exception as e:
        print(f"\nTraining failed for {game_name}: {e}")
        return
    
    # Test the trained agent
    print(f"\nTesting trained agent on {game_name}...")
    try:
        test_environment = DQNAtariEnv(
            game_name=game_name,
            device=device,
            render_mode='human',
            repeat=game_config['repeat'],
            no_ops=game_config['no_ops'],
            fire_first=game_config['fire_first']
        )
        agent.test(env=test_environment)
    except Exception as e:
        print(f"Testing failed: {e}")
    
    print(f"\nAll done! Check the models/ directory for saved models.")


if __name__ == "__main__":
    main()