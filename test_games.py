

from atari_env import get_available_atari_games, get_game_config, DQNAtariEnv
import torch


def test_available_games():
    print("TESTING AVAILABLE ATARI GAMES")
    print("=" * 50)
    
    # Get available games
    available_games = get_available_atari_games()
    
    if not available_games:
        print("No Atari games found!")
        return
    
    print(f"Found {len(available_games)} available Atari games:")
    print()
    
    # Test each game
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_games = []
    
    for i, game in enumerate(available_games, 1):
        print(f"{i:2d}. Testing {game:<20}", end="")
        
        try:
            # Get game config
            config = get_game_config(game)
            
            # Try to create environment
            env = DQNAtariEnv(
                game_name=game,
                device=device,
                repeat=config['repeat'],
                no_ops=config['no_ops'],
                fire_first=config['fire_first']
            )
            
            nb_actions = env.get_nb_actions()
            working_games.append((game, nb_actions, config))
            print(f"({nb_actions} actions)")
            
            # Clean up
            env.close()
            
        except Exception as e:
            print(f"({str(e)[:30]}...)")
    
    print(f"\nSUMMARY:")
    print(f"Total games tested: {len(available_games)}")
    print(f"Working games: {len(working_games)}")
    
    if working_games:
        print(f"\nRECOMMENDED GAMES FOR TRAINING:")
        print("-" * 40)
        
        # Sort by number of actions (simpler games are easier to train)
        working_games.sort(key=lambda x: x[1])
        
        for game, nb_actions, config in working_games[:10]:  # Show top 10
            print(f"• {game:<20} ({nb_actions} actions) - Fire first: {'Yes' if config['fire_first'] else 'No'}")


def test_specific_game(game_name):
    print(f"TESTING SPECIFIC GAME: {game_name}")
    print("=" * 40)
    
    try:
        # Get game config
        config = get_game_config(game_name)
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Create environment
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        env = DQNAtariEnv(
            game_name=game_name,
            device=device,
            repeat=config['repeat'],
            no_ops=config['no_ops'],
            fire_first=config['fire_first']
        )
        
        print(f"\nEnvironment created successfully!")
        print(f"Number of actions: {env.get_nb_actions()}")
        
        # Test reset
        state = env.reset()
        print(f"State shape: {state.shape}")
        
        # Test a few steps
        print(f"\nTesting gameplay...")
        for step in range(5):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(f"  Step {step+1}: Action={action}, Reward={reward.item():.2f}, Done={done.item()}")
            if done.item():
                break
        
        env.close()
        print(f"\nGame {game_name} is working correctly!")
        
    except Exception as e:
        print(f"Failed to test {game_name}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific game
        game_name = sys.argv[1]
        test_specific_game(game_name)
    else:
        # Test all games
        test_available_games() 