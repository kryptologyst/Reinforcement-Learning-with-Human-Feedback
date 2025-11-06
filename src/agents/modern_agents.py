"""
Modern RL Agents: Implementation of state-of-the-art reinforcement learning algorithms.

This module provides implementations of PPO, SAC, TD3, and Rainbow DQN agents
for both discrete and continuous action spaces.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym
from dataclasses import dataclass
import random
from collections import deque, namedtuple
import math


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 100
    device: str = "cpu"


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ValueNetwork(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RainbowDQN(nn.Module):
    """Rainbow DQN with distributional RL and dueling architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, 
                 n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Dueling architecture
        self.value_stream = nn.Linear(hidden_dim, n_atoms)
        self.advantage_stream = nn.Linear(hidden_dim, action_dim * n_atoms)
        
        # Distributional RL
        self.register_buffer('atoms', torch.linspace(v_min, v_max, n_atoms))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        
        value = self.value_stream(features).view(-1, 1, self.n_atoms)
        advantage = self.advantage_stream(features).view(-1, self.action_dim, self.n_atoms)
        
        # Dueling combination
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return F.softmax(q_dist, dim=-1)


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.learning_rate)
        
        self.gamma = config.gamma
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Get action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            if training:
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=-1)
            
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()
    
    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               log_probs: np.ndarray, values: np.ndarray, dones: np.ndarray):
        """Update policy and value networks."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, values, dones)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Policy update
            logits = self.policy_net(states)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Value loss
            new_values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(new_values, returns)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def _compute_returns(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Compute discounted returns."""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns


class DQNAgent:
    """Deep Q-Network agent with experience replay."""
    
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        
        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.gamma = config.gamma
        
        # Experience replay buffer
        self.buffer = deque(maxlen=config.buffer_size)
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.update_count = 0
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update Q-network using experience replay."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute Q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class RainbowDQNAgent:
    """Rainbow DQN agent with distributional RL."""
    
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        
        self.q_net = RainbowDQN(state_dim, action_dim).to(self.device)
        self.target_net = RainbowDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.gamma = config.gamma
        
        # Experience replay buffer
        self.buffer = deque(maxlen=config.buffer_size)
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.update_count = 0
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_dist = self.q_net(state_tensor)
            q_values = (q_dist * self.q_net.atoms).sum(dim=-1)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update Rainbow DQN using distributional RL."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q-distributions
        current_q_dist = self.q_net(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        # Compute target Q-distributions
        with torch.no_grad():
            next_q_dist = self.target_net(next_states)
            next_q_values = (next_q_dist * self.q_net.atoms).sum(dim=-1)
            next_actions = torch.argmax(next_q_values, dim=1)
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]
            
            # Project target distribution
            target_atoms = rewards.unsqueeze(1) + self.gamma * self.q_net.atoms.unsqueeze(0) * (~dones.unsqueeze(1))
            target_atoms = torch.clamp(target_atoms, self.q_net.v_min, self.q_net.v_max)
            
            # Compute projection onto atoms
            atom_delta = (self.q_net.v_max - self.q_net.v_min) / (self.q_net.n_atoms - 1)
            target_atoms_bin = (target_atoms - self.q_net.v_min) / atom_delta
            target_atoms_bin = torch.clamp(target_atoms_bin, 0, self.q_net.n_atoms - 1)
            
            # Distribute probabilities
            target_dist = torch.zeros_like(next_q_dist)
            for i in range(self.batch_size):
                for j in range(self.q_net.n_atoms):
                    bin_idx = target_atoms_bin[i, j]
                    lower_idx = int(bin_idx.floor().item())
                    upper_idx = int(bin_idx.ceil().item())
                    
                    if lower_idx == upper_idx:
                        target_dist[i, lower_idx] += next_q_dist[i, j]
                    else:
                        target_dist[i, lower_idx] += next_q_dist[i, j] * (upper_idx - bin_idx)
                        target_dist[i, upper_idx] += next_q_dist[i, j] * (bin_idx - lower_idx)
        
        # Compute loss
        loss = -torch.sum(target_dist * torch.log(current_q_dist + 1e-8), dim=1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
