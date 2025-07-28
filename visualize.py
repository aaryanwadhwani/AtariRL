
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys
import glob
import re
from io import BytesIO
from typing import Optional, List, Tuple

from model import AtariNet
from atari_env import DQNAtariEnv, get_available_atari_games, get_nb_actions_for_game
from utils import find_latest_checkpoint, select_model_for_game, get_model_info, ensure_directory


class AtariVisualizer:
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.media_dir = "Media"
        ensure_directory(self.media_dir)
    
    def plot_conv1_filters(self, checkpoint_path: str, game_name: Optional[str] = None) -> str:

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Auto-detect game name if not provided
        if game_name is None:
            game_name, _ = get_model_info(checkpoint_path)
            if game_name == "Unknown":
                game_name = "Breakout"  # Default fallback
        
        nb_actions = get_nb_actions_for_game(game_name)
        print(f"Game: {game_name}")
        print(f"Number of actions: {nb_actions}")
        
        # Load model
        net = AtariNet(nb_actions=nb_actions).to(self.device)
        try:
            checkpoint_data = net.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint: {os.path.basename(checkpoint_path)}")
            if 'game_name' in checkpoint_data:
                game_name = checkpoint_data['game_name']
            if 'epoch' in checkpoint_data:
                print(f"Epoch: {checkpoint_data['epoch']}")
        except:
            # Fallback to old weight-only format
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            net.load_state_dict(state_dict)
            print(f"Loaded weights: {os.path.basename(checkpoint_path)}")
        
        net.eval()
        
        # Extract filters
        filters = net.conv1.weight.detach().cpu().numpy()
        n_filters = filters.shape[0]
        ncols = 8
        nrows = (n_filters + ncols - 1) // ncols
        
        print(f"Found {n_filters} filters in first layer")
        
        # Create plot
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.5, nrows*1.5))
        for idx, ax in enumerate(axes.flat):
            if idx < n_filters:
                fmap = filters[idx, 0]  # Single channel
                ax.imshow(fmap, interpolation='nearest', cmap='viridis')
            ax.axis('off')
        
        plt.suptitle(f"Conv1 Filters - {game_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save image
        output_path = os.path.join(self.media_dir, f"{game_name}_conv1_filters.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved filters to {output_path}")
        plt.show()
        
        return output_path
    
    def plot_layer_activations(self, checkpoint_path: str, game_name: Optional[str] = None, 
                              layer_name: str = "conv1") -> str:

        if game_name is None:
            game_name, _ = get_model_info(checkpoint_path)
            if game_name == "Unknown":
                game_name = "Breakout"
        
        nb_actions = get_nb_actions_for_game(game_name)
        
        # Load model
        net = AtariNet(nb_actions=nb_actions).to(self.device)
        try:
            net.load_checkpoint(checkpoint_path)
        except:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            net.load_state_dict(state_dict)
        
        net.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 84, 84).to(self.device)
        
        # Get activations
        activations = None
        def hook_fn(module, input, output):
            nonlocal activations
            activations = output.detach().cpu().numpy()
        
        # Register hook
        if layer_name == "conv1":
            hook = net.conv1.register_forward_hook(hook_fn)
        elif layer_name == "conv2":
            hook = net.conv2.register_forward_hook(hook_fn)
        elif layer_name == "conv3":
            hook = net.conv3.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        # Forward pass
        with torch.no_grad():
            net(dummy_input)
        
        hook.remove()
        
        # Plot activations
        n_channels = activations.shape[1]
        ncols = 8
        nrows = (n_channels + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.5, nrows*1.5))
        for idx, ax in enumerate(axes.flat):
            if idx < n_channels:
                activation = activations[0, idx]
                ax.imshow(activation, cmap='viridis')
            ax.axis('off')
        
        plt.suptitle(f"{layer_name.upper()} Activations - {game_name}", fontsize=16)
        plt.tight_layout()
        
        # Save image
        output_path = os.path.join(self.media_dir, f"{layer_name}_activations.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved {layer_name} activations to {output_path}")
        plt.show()
        
        return output_path
    
    def create_filter_animation(self, game_name: Optional[str] = None, fps: int = 2) -> str:

        model_dir = "models"
        
        if game_name:
            # Game-specific animation
            files = glob.glob(os.path.join(model_dir, f"*{game_name}*.pt"))
            output_path = os.path.join(self.media_dir, f"{game_name}_filter_evolution.gif")
        else:
            # All models animation
            files = glob.glob(os.path.join(model_dir, "*.pt"))
            output_path = os.path.join(self.media_dir, "filter_evolution.gif")
        
        if not files:
            raise FileNotFoundError(f"No model files found for {game_name or 'any game'}")
        
        # Sort by epoch
        def epoch_key(f):
            _, epoch = get_model_info(f)
            return epoch
        
        files = sorted(files, key=epoch_key)
        print(f"Found {len(files)} model files")
        
        # Process each checkpoint
        images = []
        for ckpt in files:
            try:
                ckpt_game_name, epoch = get_model_info(ckpt)
                if game_name and ckpt_game_name != game_name:
                    continue
                
                nb_actions = get_nb_actions_for_game(ckpt_game_name)
                
                # Load model
                net = AtariNet(nb_actions=nb_actions).cpu()
                try:
                    net.load_checkpoint(ckpt)
                except:
                    sd = torch.load(ckpt, map_location="cpu")
                    net.load_state_dict(sd)
                
                net.eval()
                
                # Extract filters
                W = net.conv1.weight.detach().numpy()[:, 0]
                n_filters, kH, kW = W.shape
                ncols = 8
                nrows = (n_filters + ncols - 1) // ncols
                
                # Create frame
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.2, nrows*1.2))
                for i, ax in enumerate(axes.flat):
                    ax.axis("off")
                    if i < n_filters:
                        ax.imshow(W[i], interpolation="nearest", cmap="viridis")
                
                plt.suptitle(f"Conv1 Filters - {ckpt_game_name} @ Epoch {epoch}", fontsize=12)
                plt.tight_layout()
                
                # Convert to image
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                img = imageio.imread(buf)
                images.append(img)
                plt.close(fig)
                
                print(f"Processed {os.path.basename(ckpt)}")
                
            except Exception as e:
                print(f"Error processing {ckpt}: {e}")
                continue
        
        if not images:
            raise RuntimeError("No valid model files could be processed")
        
        # Save GIF
        imageio.mimsave(output_path, images, duration=1000//fps)
        print(f"Saved filter evolution to {output_path}")
        print(f"Created animation with {len(images)} frames")
        
        return output_path
    
    def create_saliency_overlay(self, checkpoint_path: str, game_name: str = "Breakout", 
                               n_frames: int = 50, duration: float = 0.1) -> str:

        nb_actions = get_nb_actions_for_game(game_name)
        
        # Load model
        net = AtariNet(nb_actions=nb_actions).to(self.device)
        try:
            net.load_checkpoint(checkpoint_path)
        except:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            net.load_state_dict(state_dict)
        net.eval()
        
        # Set up environment
        env = DQNAtariEnv(game_name=game_name, device=self.device)
        state = env.reset()
        
        frames = []
        for _ in range(n_frames):
            # Make state require grad for saliency
            state = state.clone().detach().requires_grad_(True)
            
            # Forward pass and get action
            qvals = net(state)
            action = qvals.argmax(dim=1).item()
            
            # Backprop to get saliency
            net.zero_grad()
            qvals[0, action].backward()
            saliency = state.grad[0, 0].abs().cpu().numpy()
            
            # Normalize saliency
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            # Create heatmap
            heatmap = plt.cm.jet(saliency)[..., :3]
            
            # Convert frame to RGB
            frame_gray = state[0, 0].detach().cpu().numpy()
            frame_rgb = np.stack([frame_gray] * 3, axis=-1)
            
            # Blend frame with heatmap
            overlay = frame_rgb * 0.6 + heatmap * 0.4
            overlay = np.clip(overlay, 0, 1)
            
            # Upscale
            overlay = (overlay * 255).astype(np.uint8)
            overlay = np.kron(overlay, np.ones((4, 4, 1), dtype=np.uint8))
            
            frames.append(overlay)
            
            # Step environment
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
        
        env.close()
        
        # Save GIF
        output_path = os.path.join(self.media_dir, "saliency_overlay.gif")
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"Saved saliency overlay to {output_path}")
        
        return output_path
    
    def export_to_onnx(self, checkpoint_path: str, game_name: Optional[str] = None) -> str:

        if game_name is None:
            game_name, _ = get_model_info(checkpoint_path)
            if game_name == "Unknown":
                game_name = "Breakout"
        
        nb_actions = get_nb_actions_for_game(game_name)
        
        # Load model
        net = AtariNet(nb_actions=nb_actions).to(self.device)
        try:
            net.load_checkpoint(checkpoint_path)
        except:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            net.load_state_dict(state_dict)
        
        net.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 84, 84).to(self.device)
        
        # Export to ONNX
        output_path = os.path.join(self.media_dir, "atari_dqn.onnx")
        torch.onnx.export(
            net,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Exported model to {output_path}")
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Atari DQN Visualizer")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--game", type=str, help="Game name")
    parser.add_argument("--type", type=str, choices=["filters", "activations", "animation", "saliency", "onnx"], 
                       default="filters", help="Visualization type")
    parser.add_argument("--layer", type=str, default="conv1", help="Layer for activations (conv1, conv2, conv3)")
    parser.add_argument("--fps", type=int, default=2, help="FPS for animations")
    
    args = parser.parse_args()
    
    visualizer = AtariVisualizer()
    
    # Auto-select checkpoint if not provided
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        game_name = args.game or "Breakout"
        checkpoint_path = select_model_for_game(game_name)
        if checkpoint_path is None:
            print("No suitable model found")
            return
    
    try:
        if args.type == "filters":
            visualizer.plot_conv1_filters(checkpoint_path, args.game)
        elif args.type == "activations":
            visualizer.plot_layer_activations(checkpoint_path, args.game, args.layer)
        elif args.type == "animation":
            visualizer.create_filter_animation(args.game, args.fps)
        elif args.type == "saliency":
            visualizer.create_saliency_overlay(checkpoint_path, args.game or "Breakout")
        elif args.type == "onnx":
            visualizer.export_to_onnx(checkpoint_path, args.game)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 