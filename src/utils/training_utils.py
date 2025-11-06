"""
Training Utilities: Configuration, logging, and training helpers.

This module provides utilities for training RL agents including configuration
management, logging, checkpointing, and visualization.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import json
import logging
import os
import pickle
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import wandb
from tensorboard import SummaryWriter


@dataclass
class TrainingConfig:
    """Configuration for training RL agents."""
    # Environment settings
    env_name: str = "rlhf"
    env_config: Dict[str, Any] = None
    
    # Agent settings
    agent_type: str = "ppo"  # ppo, dqn, rainbow_dqn
    agent_config: Dict[str, Any] = None
    
    # Training settings
    total_timesteps: int = 100000
    eval_freq: int = 10000
    save_freq: int = 50000
    log_freq: int = 1000
    
    # Evaluation settings
    n_eval_episodes: int = 10
    
    # Logging settings
    log_dir: str = "logs"
    use_wandb: bool = False
    use_tensorboard: bool = True
    project_name: str = "rlhf_project"
    
    # Device settings
    device: str = "cpu"
    
    def __post_init__(self):
        if self.env_config is None:
            self.env_config = {}
        if self.agent_config is None:
            self.agent_config = {}


class Logger:
    """Unified logger for training metrics and visualization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup TensorBoard
        if config.use_tensorboard:
            self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
        else:
            self.tb_writer = None
        
        # Setup Weights & Biases
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=asdict(config),
                dir=str(self.log_dir)
            )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.training_losses = []
        
    def setup_logging(self):
        """Setup file and console logging."""
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_episode(self, episode: int, reward: float, length: int, info: Dict = None):
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if episode % self.config.log_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            
            self.logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}, "
                           f"Avg Reward (100)={avg_reward:.2f}, Avg Length (100)={avg_length:.2f}")
            
            if self.tb_writer:
                self.tb_writer.add_scalar("Episode/Reward", reward, episode)
                self.tb_writer.add_scalar("Episode/Length", length, episode)
                self.tb_writer.add_scalar("Episode/AvgReward100", avg_reward, episode)
                self.tb_writer.add_scalar("Episode/AvgLength100", avg_length, episode)
            
            if self.config.use_wandb:
                wandb.log({
                    "episode": episode,
                    "reward": reward,
                    "length": length,
                    "avg_reward_100": avg_reward,
                    "avg_length_100": avg_length
                })
    
    def log_evaluation(self, episode: int, rewards: List[float], lengths: List[int]):
        """Log evaluation metrics."""
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_length = np.mean(lengths)
        
        self.eval_rewards.append(mean_reward)
        self.eval_lengths.append(mean_length)
        
        self.logger.info(f"Evaluation at episode {episode}: "
                        f"Mean Reward={mean_reward:.2f}±{std_reward:.2f}, "
                        f"Mean Length={mean_length:.2f}")
        
        if self.tb_writer:
            self.tb_writer.add_scalar("Evaluation/MeanReward", mean_reward, episode)
            self.tb_writer.add_scalar("Evaluation/StdReward", std_reward, episode)
            self.tb_writer.add_scalar("Evaluation/MeanLength", mean_length, episode)
        
        if self.config.use_wandb:
            wandb.log({
                "episode": episode,
                "eval_mean_reward": mean_reward,
                "eval_std_reward": std_reward,
                "eval_mean_length": mean_length
            })
    
    def log_training_loss(self, episode: int, loss: float, loss_type: str = "total"):
        """Log training loss."""
        self.training_losses.append(loss)
        
        if self.tb_writer:
            self.tb_writer.add_scalar(f"Training/{loss_type}Loss", loss, episode)
        
        if self.config.use_wandb:
            wandb.log({
                "episode": episode,
                f"{loss_type}_loss": loss
            })
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) > 100:
            window = min(100, len(self.episode_rewards) // 10)
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), smoothed, color='red', linewidth=2)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.3, color='green')
        if len(self.episode_lengths) > 100:
            window = min(100, len(self.episode_lengths) // 10)
            smoothed = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.episode_lengths)), smoothed, color='red', linewidth=2)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True)
        
        # Evaluation rewards
        if self.eval_rewards:
            axes[1, 0].plot(self.eval_rewards, 'o-', color='purple')
            axes[1, 0].set_title("Evaluation Rewards")
            axes[1, 0].set_xlabel("Evaluation")
            axes[1, 0].set_ylabel("Mean Reward")
            axes[1, 0].grid(True)
        
        # Training losses
        if self.training_losses:
            axes[1, 1].plot(self.training_losses, color='orange')
            axes[1, 1].set_title("Training Losses")
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def close(self):
        """Close logger and cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.config.use_wandb:
            wandb.finish()


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, agent: Any, episode: int, config: TrainingConfig, 
                       additional_data: Dict = None) -> str:
        """Save agent checkpoint."""
        checkpoint_data = {
            "episode": episode,
            "config": asdict(config),
            "agent_state": agent.__dict__ if hasattr(agent, '__dict__') else None,
            "additional_data": additional_data or {}
        }
        
        # Save PyTorch models
        if hasattr(agent, 'policy_net') and hasattr(agent.policy_net, 'state_dict'):
            checkpoint_data["policy_state_dict"] = agent.policy_net.state_dict()
        if hasattr(agent, 'value_net') and hasattr(agent.value_net, 'state_dict'):
            checkpoint_data["value_state_dict"] = agent.value_net.state_dict()
        if hasattr(agent, 'q_net') and hasattr(agent.q_net, 'state_dict'):
            checkpoint_data["q_net_state_dict"] = agent.q_net.state_dict()
        if hasattr(agent, 'target_net') and hasattr(agent.target_net, 'state_dict'):
            checkpoint_data["target_net_state_dict"] = agent.target_net.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load agent checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint path."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_episode_*.pkl"))
        if not checkpoints:
            return None
        
        # Sort by episode number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoints[-1])


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, config_path: str):
    """Save training configuration to YAML or JSON file."""
    config_path = Path(config_path)
    config_dict = asdict(config)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_default_config() -> TrainingConfig:
    """Create a default training configuration."""
    return TrainingConfig(
        env_name="rlhf",
        agent_type="ppo",
        total_timesteps=100000,
        eval_freq=10000,
        save_freq=50000,
        log_freq=1000,
        n_eval_episodes=10,
        log_dir="logs",
        use_wandb=False,
        use_tensorboard=True,
        project_name="rlhf_project",
        device="cpu"
    )
