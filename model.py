import torch
import torch.nn as nn


class AtariNet(nn.Module):

    def __init__(self, nb_actions=4):

        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.action_value1 = nn.Linear(3136, 1024)

        self.action_value2 = nn.Linear(1024, 1024)

        self.action_value3 = nn.Linear(1024, nb_actions)

        self.state_value1 = nn.Linear(3136, 1024)

        self.state_value2 = nn.Linear(1024, 1024)

        self.state_value3 = nn.Linear(1024, 1)


    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value3(action_value))

        output = state_value + (action_value - action_value.mean())

        return output

    def save_checkpoint(self, checkpoint_path, optimizer=None, epoch=None, epsilon=None, stats=None, game_name=None, game_config=None):

        checkpoint = {
            'model_state_dict': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if epsilon is not None:
            checkpoint['epsilon'] = epsilon
        if stats is not None:
            checkpoint['stats'] = stats
        if game_name is not None:
            checkpoint['game_name'] = game_name
        if game_config is not None:
            checkpoint['game_config'] = game_config
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, optimizer=None):

        # PyTorch 2.6+: weights_only=False is required for full checkpoint loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")
        out = {}
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded optimizer state from {checkpoint_path}")
        for key in ['epoch', 'epsilon', 'stats', 'game_name', 'game_config']:
            if key in checkpoint:
                out[key] = checkpoint[key]
        return out

    def save_the_model(self, weights_filename='models/latest.pt'):
        # For backward compatibility: save only weights
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")