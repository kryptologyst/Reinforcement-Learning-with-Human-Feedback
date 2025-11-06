"""
Main Training Script: Unified training interface for RL agents.

This script provides a unified interface for training different RL agents
on various environments with comprehensive logging and visualization.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import gymnasium as gym

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.envs.rlhf_env import RLHFEnvironment, create_default_rlhf_env
from src.envs.mock_envs import GridWorld, SimpleMountainCar, create_default_gridworld, create_default_mountaincar
from src.agents.modern_agents import PPOAgent, DQNAgent, RainbowDQNAgent, AgentConfig
from src.utils.training_utils import (
    TrainingConfig, Logger, CheckpointManager, 
    load_config, save_config, create_default_config
)


def create_environment(config: TrainingConfig) -> gym.Env:
    """Create environment based on configuration."""
    if config.env_name == "rlhf":
        return create_default_rlhf_env()
    elif config.env_name == "gridworld":
        return create_default_gridworld()
    elif config.env_name == "mountaincar":
        return create_default_mountaincar()
    elif config.env_name == "cartpole":
        return gym.make("CartPole-v1")
    else:
        raise ValueError(f"Unknown environment: {config.env_name}")


def create_agent(env: gym.Env, config: TrainingConfig):
    """Create agent based on configuration."""
    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.n
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    
    agent_config = AgentConfig(**config.agent_config)
    
    if config.agent_type == "ppo":
        return PPOAgent(state_dim, action_dim, agent_config)
    elif config.agent_type == "dqn":
        return DQNAgent(state_dim, action_dim, agent_config)
    elif config.agent_type == "rainbow_dqn":
        return RainbowDQNAgent(state_dim, action_dim, agent_config)
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


def evaluate_agent(agent, env: gym.Env, n_episodes: int = 10) -> tuple:
    """Evaluate agent performance."""
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if hasattr(agent, 'get_action'):
                action, _ = agent.get_action(obs, training=False)
            else:
                action = agent.predict(obs, deterministic=True)[0]
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return episode_rewards, episode_lengths


def train_agent(agent, env: gym.Env, config: TrainingConfig, logger: Logger, 
                checkpoint_manager: CheckpointManager):
    """Train the agent."""
    episode = 0
    total_timesteps = 0
    
    # Training data collection
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    
    while total_timesteps < config.total_timesteps:
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Episode data collection
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []
        episode_values = []
        episode_dones = []
        
        while not done:
            if config.agent_type == "ppo":
                action, log_prob = agent.get_action(obs, training=True)
                value = agent.value_net(torch.FloatTensor(obs).unsqueeze(0).to(agent.device)).item()
                
                episode_states.append(obs)
                episode_actions.append(action)
                episode_log_probs.append(log_prob)
                episode_values.append(value)
            else:
                action = agent.get_action(obs, training=True)
                episode_states.append(obs)
                episode_actions.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_dones.append(done)
            
            # Store transition for DQN agents
            if config.agent_type in ["dqn", "rainbow_dqn"]:
                agent.store_transition(episode_states[-1], episode_actions[-1], 
                                     reward, obs, done)
        
        # Log episode
        logger.log_episode(episode, episode_reward, episode_length)
        
        # Update agent
        if config.agent_type == "ppo":
            # Collect data for PPO update
            states.extend(episode_states)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            log_probs.extend(episode_log_probs)
            values.extend(episode_values)
            dones.extend(episode_dones)
            
            # Update PPO every few episodes
            if len(states) >= config.agent_config.get('batch_size', 64):
                agent.update(np.array(states), np.array(actions), np.array(rewards),
                           np.array(log_probs), np.array(values), np.array(dones))
                states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        elif config.agent_type in ["dqn", "rainbow_dqn"]:
            # Update DQN agents
            for _ in range(4):  # Multiple updates per episode
                agent.update()
        
        total_timesteps += episode_length
        episode += 1
        
        # Evaluation
        if episode % config.eval_freq == 0:
            eval_rewards, eval_lengths = evaluate_agent(agent, env, config.n_eval_episodes)
            logger.log_evaluation(episode, eval_rewards, eval_lengths)
        
        # Save checkpoint
        if episode % config.save_freq == 0:
            checkpoint_path = checkpoint_manager.save_checkpoint(agent, episode, config)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final evaluation
    print("Training completed. Running final evaluation...")
    eval_rewards, eval_lengths = evaluate_agent(agent, env, config.n_eval_episodes)
    logger.log_evaluation(episode, eval_rewards, eval_lengths)
    
    # Plot learning curves
    logger.plot_learning_curves()
    
    return agent


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agents")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="rlhf", 
                       choices=["rlhf", "gridworld", "mountaincar", "cartpole"],
                       help="Environment to train on")
    parser.add_argument("--agent", type=str, default="ppo",
                       choices=["ppo", "dqn", "rainbow_dqn"],
                       help="Agent type to train")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for logs")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--use-tensorboard", action="store_true", default=True,
                       help="Use TensorBoard for logging")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        config.env_name = args.env
        config.agent_type = args.agent
        config.total_timesteps = args.timesteps
        config.device = args.device
        config.log_dir = args.log_dir
        config.use_wandb = args.use_wandb
        config.use_tensorboard = args.use_tensorboard
    
    # Set device
    if config.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    config.device = device
    
    print(f"Training configuration:")
    print(f"  Environment: {config.env_name}")
    print(f"  Agent: {config.agent_type}")
    print(f"  Device: {config.device}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Log directory: {config.log_dir}")
    
    # Create environment
    env = create_environment(config)
    print(f"Environment created: {env}")
    
    # Create agent
    agent = create_agent(env, config)
    print(f"Agent created: {config.agent_type}")
    
    # Create logger and checkpoint manager
    logger = Logger(config)
    checkpoint_manager = CheckpointManager()
    
    # Train agent
    try:
        trained_agent = train_agent(agent, env, config, logger, checkpoint_manager)
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
        
    finally:
        # Cleanup
        logger.close()
        env.close()
    
    return trained_agent


if __name__ == "__main__":
    main()
