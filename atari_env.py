import collections
import cv2
import gym
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from PIL import Image
import torch


class DQNAtariEnv(gym.Wrapper):

    def __init__(self, game_name, render_mode='rgb_array', repeat=4, no_ops=0,
                 fire_first=False, device='cpu'):

        # Construct the full environment name
        env_name = f"{game_name}NoFrameskip-v4"
        
        try:
            env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            # Try without NoFrameskip if the above fails
            try:
                env_name = f"{game_name}-v4"
                env = gym.make(env_name, render_mode=render_mode)
            except Exception as e2:
                raise ValueError(f"Could not create environment for game '{game_name}'. "
                               f"Available games can be found using get_available_atari_games(). "
                               f"Error: {e2}")

        super(DQNAtariEnv, self).__init__(env)

        self.game_name = game_name
        self.repeat = repeat
        self.image_shape = (84, 84)
        self.frame_buffer = []
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.device = device
        
        # Get the number of actions for this game
        self.nb_actions = env.action_space.n
        
        # Initialize lives tracking (for games that have lives)
        try:
            self.lives = env.ale.lives()
            self.has_lives = True
        except:
            self.lives = 0
            self.has_lives = False

    def step(self, action):
        total_reward = 0
        done = False

        # Repeat actions to speed up training and average over frames
        for i in range(self.repeat):
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            # Handle lives penalty for games that have lives
            if self.has_lives:
                try:
                    current_lives = info.get('lives', self.env.ale.lives())
                    if current_lives < self.lives:
                        total_reward = total_reward - 1
                        self.lives = current_lives
                except:
                    pass

            self.frame_buffer.append(observation)

            if done:
                break

        # Take max over last 2 frames to remove flickering
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return max_frame, total_reward, done, info

    def reset(self):
        self.frame_buffer = []
        observation, _ = self.env.reset()

        # Reset lives tracking
        if self.has_lives:
            try:
                self.lives = self.env.ale.lives()
            except:
                pass

        # Perform no-op actions at the start
        for _ in range(self.no_ops):
            observation, _, _, _, _ = self.env.step(0)

        # Fire first if required (for games like Breakout)
        if self.fire_first:
            observation, _, _, _, _ = self.env.step(1)

        observation = self.process_observation(observation)
        return observation

    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0
        img = img.to(self.device)
        return img

    def get_nb_actions(self):
        return self.nb_actions


def get_available_atari_games():

    # Common Atari games that work well with DQN
    common_games = [
        'Breakout', 'Pong', 'SpaceInvaders', 'Qbert', 'MsPacman',
        'Asteroids', 'BeamRider', 'Boxing', 'ChopperCommand', 'CrazyClimber',
        'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway',
        'Frostbite', 'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond',
        'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'NameThisGame',
        'Phoenix', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert',
        'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
        'Solaris', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham',
        'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge',
        'Zaxxon'
    ]
    
    # Filter to only include games that actually exist in the environment
    available_games = []
    for game in common_games:
        try:
            env_name = f"{game}NoFrameskip-v4"
            gym.make(env_name)
            available_games.append(game)
        except:
            try:
                env_name = f"{game}-v4"
                gym.make(env_name)
                available_games.append(game)
            except:
                continue
    
    return sorted(available_games)


def get_game_config(game_name):

    # Default configuration
    config = {
        'repeat': 4,
        'no_ops': 0,
        'fire_first': False,
        'learning_rate': 0.00001,
        'epsilon': 1.0,
        'min_epsilon': 0.1,
        'nb_warmup': 5000,
        'memory_capacity': 100000,
        'batch_size': 64
    }
    
    # Game-specific configurations
    game_configs = {
        'Breakout': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Pong': {
            'no_ops': 30,
        },
        'SpaceInvaders': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Qbert': {
            'no_ops': 30,
        },
        'MsPacman': {
            'no_ops': 30,
        },
        'Asteroids': {
            'fire_first': True,
            'no_ops': 30,
        },
        'BeamRider': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Boxing': {
            'no_ops': 30,
        },
        'ChopperCommand': {
            'fire_first': True,
            'no_ops': 30,
        },
        'CrazyClimber': {
            'no_ops': 30,
        },
        'DemonAttack': {
            'fire_first': True,
            'no_ops': 30,
        },
        'DoubleDunk': {
            'no_ops': 30,
        },
        'Enduro': {
            'no_ops': 30,
        },
        'FishingDerby': {
            'no_ops': 30,
        },
        'Freeway': {
            'no_ops': 30,
        },
        'Frostbite': {
            'no_ops': 30,
        },
        'Gopher': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Gravitar': {
            'fire_first': True,
            'no_ops': 30,
        },
        'IceHockey': {
            'no_ops': 30,
        },
        'Jamesbond': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Kangaroo': {
            'no_ops': 30,
        },
        'Krull': {
            'fire_first': True,
            'no_ops': 30,
        },
        'KungFuMaster': {
            'no_ops': 30,
        },
        'MontezumaRevenge': {
            'no_ops': 30,
        },
        'NameThisGame': {
            'no_ops': 30,
        },
        'Phoenix': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Pitfall': {
            'no_ops': 30,
        },
        'PrivateEye': {
            'no_ops': 30,
        },
        'Riverraid': {
            'fire_first': True,
            'no_ops': 30,
        },
        'RoadRunner': {
            'no_ops': 30,
        },
        'Robotank': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Seaquest': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Skiing': {
            'no_ops': 30,
        },
        'Solaris': {
            'fire_first': True,
            'no_ops': 30,
        },
        'StarGunner': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Tennis': {
            'no_ops': 30,
        },
        'TimePilot': {
            'fire_first': True,
            'no_ops': 30,
        },
        'Tutankham': {
            'no_ops': 30,
        },
        'UpNDown': {
            'no_ops': 30,
        },
        'Venture': {
            'no_ops': 30,
        },
        'VideoPinball': {
            'no_ops': 30,
        },
        'WizardOfWor': {
            'fire_first': True,
            'no_ops': 30,
        },
        'YarsRevenge': {
            'no_ops': 30,
        },
        'Zaxxon': {
            'fire_first': True,
            'no_ops': 30,
        }
    }
    
    # Update with game-specific config if available
    if game_name in game_configs:
        config.update(game_configs[game_name])
    
    return config


def get_nb_actions_for_game(game_name):

    try:
        import torch
        device = torch.device('cpu')
        env = DQNAtariEnv(game_name=game_name, device=device)
        nb_actions = env.get_nb_actions()
        env.close()
        return nb_actions
    except Exception as e:
        print(f"Could not determine actions for {game_name}: {e}")
        # Return default based on common Atari games
        return 4 