
import matplotlib.pyplot as plt
import os
import glob
import re
from typing import Optional, List, Tuple


def display_observation_image(observation):
    observation = observation.squeeze(0)
    observation = observation.squeeze(0)
    plt.imshow(observation, cmap='gray')
    plt.axis('off')
    plt.show()


def find_latest_checkpoint(game_name: str, checkpoint_dir: str = "models") -> Optional[str]:

    pattern = os.path.join(checkpoint_dir, f"model_{game_name}_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    
    def extract_epoch(f):
        m = re.search(rf"model_{re.escape(game_name)}_(\d+)\.pt", os.path.basename(f))
        return int(m.group(1)) if m else -1
    
    files = sorted(files, key=extract_epoch, reverse=True)
    return files[0] if files else None


def list_checkpoints(game_name: str, checkpoint_dir: str = "models") -> List[str]:

    pattern = os.path.join(checkpoint_dir, f"model_{game_name}_*.pt")
    files = glob.glob(pattern)
    
    def extract_epoch(f):
        m = re.search(rf"model_{re.escape(game_name)}_(\d+)\.pt", os.path.basename(f))
        return int(m.group(1)) if m else -1
    
    return sorted(files, key=extract_epoch, reverse=True)


def select_model_for_game(game_name: str, models_dir: str = "models") -> Optional[str]:

    if not os.path.exists(models_dir):
        print("No models directory found")
        return None
    
    # First, try to find the latest checkpoint for this game
    latest_checkpoint = find_latest_checkpoint(game_name, models_dir)
    if latest_checkpoint:
        print(f"Found latest checkpoint for {game_name}: {os.path.basename(latest_checkpoint)}")
        return latest_checkpoint
    
    # If no game-specific checkpoint, look for any model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not model_files:
        print("No model files found")
        return None
    
    print(f"\nNo specific checkpoint found for {game_name}")
    print("Available models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i:2d}. {model_file}")
    
    try:
        choice = input(f"\nSelect model (1-{len(model_files)}) or press Enter to skip: ").strip()
        if choice:
            choice_num = int(choice)
            if 1 <= choice_num <= len(model_files):
                return os.path.join(models_dir, model_files[choice_num - 1])
    except (ValueError, KeyboardInterrupt):
        pass
    
    return None


def get_model_info(model_path: str) -> Tuple[str, int]:

    filename = os.path.basename(model_path)
    
    # Try to extract game name from filename
    game_name = "Unknown"
    epoch = 0
    
    # Look for patterns like "game_name_epoch.pt" or "model_iter_epoch.pt"
    if "model_iter_" in filename:
        # Extract epoch number
        match = re.search(r"model_iter_(\d+)\.pt", filename)
        if match:
            epoch = int(match.group(1))
    else:
        # Try to extract game name and epoch
        match = re.search(r"([A-Za-z]+)_(\d+)\.pt", filename)
        if match:
            game_name = match.group(1)
            epoch = int(match.group(2))
    
    return game_name, epoch


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True) 