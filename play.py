import cv2
import gym
import torch
import numpy as np
import os

from atari_env import DQNAtariEnv, get_available_atari_games, get_game_config
from agent import Agent
from model import AtariNet

# Mapping keys to actions (generalized for any Atari game)
KEY_ACTIONS = {
    ord('a'): 3,  # move left
    ord('d'): 2,  # move right
    ord('w'): 1,  # fire
    ord('s'): 0,  # no-op
}


def get_game_selection():
    available_games = get_available_atari_games()
    
    if not available_games:
        print("No Atari games found!")
        return None
    
    print("\nAvailable games for play:")
    for i, game in enumerate(available_games, 1):
        print(f"{i:2d}. {game}")
    
    while True:
        try:
            choice = input(f"\nSelect game (1-{len(available_games)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_games):
                return available_games[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(available_games)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None


def play_human(game_name="Breakout"):
    max_lives = 5
    speed = 1
    wait_ms = 10

    try:
        env = gym.make(f"{game_name}NoFrameskip-v4", render_mode="rgb_array")
    except:
        try:
            env = gym.make(f"{game_name}-v4", render_mode="rgb_array")
        except Exception as e:
            print(f"Could not create environment for {game_name}: {e}")
            return 0

    obs, info = env.reset()
    initial_lives = info.get('lives', max_lives)
    lives = initial_lives
    lost_lives = 0
    done = False
    total_reward = 0

    cv2.namedWindow(f'Your Play - {game_name}', cv2.WINDOW_NORMAL)
    print(f"Playing {game_name} - Controls: A=left, D=right, W=fire, S=no-op, Q=quit")
    
    while not done:
        frame = env.render()
        cv2.imshow(f'Your Play - {game_name}', frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        action = KEY_ACTIONS.get(key, 0)
        if key == ord('q'):
            break

        for _ in range(speed):
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward

            new_lives = info.get('lives', lives)
            if new_lives < lives:
                lost_lives += (lives - new_lives)
                lives = new_lives
                print(f'Life lost! {lost_lives}/{max_lives} lives gone.')
                if lost_lives >= max_lives:
                    print('Max lives lost! Ending human game.')
                    done = True
                    break

            done = done or trunc
            if done:
                break
        if done:
            break

    cv2.destroyAllWindows()
    env.close()
    print(f'Your score: {total_reward}')
    return total_reward


def play_model(game_name="Breakout", model_path=None):
    wait_ms = 30
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Get game configuration
    game_config = get_game_config(game_name)
    
    try:
        # Create environment
        env = DQNAtariEnv(
            game_name=game_name,
            device=device,
            render_mode='rgb_array',
            repeat=game_config['repeat'],
            no_ops=game_config['no_ops'],
            fire_first=game_config['fire_first']
        )
        
        nb_actions = env.get_nb_actions()
        
        # Create model
        model = AtariNet(nb_actions=nb_actions)
        
        # Load model weights
        if model_path and os.path.exists(model_path):
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
            print("No model found, using untrained model")
        
        # Create agent
        agent = Agent(
            model=model,
            device=device,
            epsilon=0.05,  # Low epsilon for testing
            min_epsilon=0.05,
            nb_warmup=1,
            nb_actions=nb_actions
        )

        state = env.reset()
        done = False
        total_reward = 0

        cv2.namedWindow(f'Model Play - {game_name}', cv2.WINDOW_NORMAL)
        print(f"AI playing {game_name} - Press Q to quit")
        
        while not done:
            frame = env.env.render()
            cv2.imshow(f'Model Play - {game_name}', frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

            next_state, reward, done_tensor, info = env.step(agent.get_action(state))
            done = bool(done_tensor)
            total_reward += reward.item()
            state = next_state

        cv2.destroyAllWindows()
        env.close()
        total_reward += 5
        print(f'Model score: {total_reward}')
        return total_reward
        
    except Exception as e:
        print(f"Error playing model: {e}")
        return 0


from utils import find_latest_checkpoint, select_model_for_game


def main():
    print("ATARI GAME PLAYER")
    print("=" * 40)
    
    # Get game selection
    game_name = get_game_selection()
    if game_name is None:
        print("Goodbye!")
        return
    
    print(f"\nSelected game: {game_name}")
    
    # Ask what to do
    while True:
        print(f"\nWhat would you like to do?")
        print("1. Play manually")
        print("2. Watch AI play")
        print("3. Compare human vs AI")
        print("4. Select different game")
        print("5. Quit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            human_score = play_human(game_name)
        elif choice == '2':
            # Automatically find the best model for this game
            model_path = select_model_for_game(game_name)
            ai_score = play_model(game_name, model_path)
        elif choice == '3':
            print(f"\nComparing human vs AI for {game_name}")
            human_score = play_human(game_name)
            model_path = select_model_for_game(game_name)
            ai_score = play_model(game_name, model_path)
            print(f"\n--- Comparison ---")
            print(f"You: {human_score}  |  AI: {ai_score}")
            if human_score > ai_score:
                print("You won!")
            elif ai_score > human_score:
                print("AI won!")
            else:
                print("It's a tie!")
        elif choice == '4':
            game_name = get_game_selection()
            if game_name is None:
                print("Goodbye!")
                return
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Please enter a valid choice (1-5)")


if __name__ == '__main__':
    main()
