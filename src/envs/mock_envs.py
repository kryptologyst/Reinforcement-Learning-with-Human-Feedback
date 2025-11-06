"""
Mock Environments: Simple environments for testing and demonstration.

This module provides simple mock environments including CartPole, GridWorld,
and MountainCar for testing RL algorithms.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass


@dataclass
class GridWorldConfig:
    """Configuration for GridWorld environment."""
    width: int = 8
    height: int = 8
    start_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (7, 7)
    obstacles: List[Tuple[int, int]] = None
    max_steps: int = 100
    reward_goal: float = 10.0
    reward_step: float = -0.1
    reward_obstacle: float = -1.0


class GridWorld(gym.Env):
    """
    Simple GridWorld environment for testing RL algorithms.
    
    The agent navigates a grid to reach a goal while avoiding obstacles.
    """
    
    def __init__(self, config: GridWorldConfig):
        super().__init__()
        self.config = config
        self.width = config.width
        self.height = config.height
        self.start_pos = config.start_pos
        self.goal_pos = config.goal_pos
        self.obstacles = config.obstacles or []
        self.max_steps = config.max_steps
        
        # Action space: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position (x, y)
        self.observation_space = spaces.Box(
            low=0, high=max(self.width, self.height), shape=(2,), dtype=np.float32
        )
        
        self.current_pos = None
        self.step_count = 0
        
        # Action mappings
        self.action_to_delta = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_pos = self.start_pos
        self.step_count = 0
        
        observation = np.array(self.current_pos, dtype=np.float32)
        info = {"position": self.current_pos}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_pos is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Get movement delta
        dx, dy = self.action_to_delta[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Check bounds
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            new_pos = (new_x, new_y)
            
            # Check obstacles
            if new_pos in self.obstacles:
                reward = self.config.reward_obstacle
                new_pos = self.current_pos  # Stay in place
            else:
                self.current_pos = new_pos
                reward = self.config.reward_step
        else:
            reward = self.config.reward_step  # Hit wall, stay in place
        
        # Check if reached goal
        terminated = self.current_pos == self.goal_pos
        if terminated:
            reward = self.config.reward_goal
        
        # Check step limit
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        observation = np.array(self.current_pos, dtype=np.float32)
        info = {
            "position": self.current_pos,
            "step_count": self.step_count,
            "reached_goal": terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if self.current_pos is None:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.width + 1):
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        for j in range(self.height + 1):
            ax.axhline(j - 0.5, color='black', linewidth=0.5)
        
        # Draw obstacles
        for obs_x, obs_y in self.obstacles:
            rect = patches.Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1, 
                                  linewidth=1, edgecolor='black', facecolor='red')
            ax.add_patch(rect)
        
        # Draw start position
        start_rect = patches.Rectangle((self.start_pos[0] - 0.5, self.start_pos[1] - 0.5), 1, 1,
                                     linewidth=2, edgecolor='black', facecolor='green')
        ax.add_patch(start_rect)
        
        # Draw goal position
        goal_rect = patches.Rectangle((self.goal_pos[0] - 0.5, self.goal_pos[1] - 0.5), 1, 1,
                                    linewidth=2, edgecolor='black', facecolor='blue')
        ax.add_patch(goal_rect)
        
        # Draw current position
        current_rect = patches.Rectangle((self.current_pos[0] - 0.4, self.current_pos[1] - 0.4), 0.8, 0.8,
                                       linewidth=2, edgecolor='black', facecolor='yellow')
        ax.add_patch(current_rect)
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"GridWorld - Step {self.step_count}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        
        plt.close(fig)
        return None
    
    def close(self):
        """Clean up environment resources."""
        pass


class SimpleMountainCar(gym.Env):
    """
    Simplified MountainCar environment for testing RL algorithms.
    
    The agent must learn to swing back and forth to build momentum
    to reach the goal at the top of the mountain.
    """
    
    def __init__(self, goal_position: float = 0.5, max_steps: int = 200):
        super().__init__()
        self.goal_position = goal_position
        self.max_steps = max_steps
        
        # Action space: 3 actions (left, nothing, right)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: position and velocity
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), dtype=np.float32
        )
        
        self.state = None
        self.step_count = 0
        
        # Action mappings
        self.action_to_force = {
            0: -1,  # Left
            1: 0,   # Nothing
            2: 1    # Right
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Random starting position
        self.state = np.array([self.np_random.uniform(-0.6, -0.4), 0], dtype=np.float32)
        self.step_count = 0
        
        info = {"position": self.state[0], "velocity": self.state[1]}
        return self.state.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        position, velocity = self.state
        
        # Apply force
        force = self.action_to_force[action]
        velocity += force * 0.001 + np.cos(3 * position) * (-0.0025)
        
        # Apply physics
        velocity = np.clip(velocity, -0.07, 0.07)
        position += velocity
        
        # Apply boundary conditions
        if position <= -1.2:
            position = -1.2
            velocity = 0
        
        self.state = np.array([position, velocity], dtype=np.float32)
        
        # Calculate reward
        reward = -1.0  # Default reward for each step
        
        # Check if reached goal
        terminated = position >= self.goal_position
        if terminated:
            reward = 0.0  # Bonus for reaching goal
        
        # Check step limit
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        info = {
            "position": position,
            "velocity": velocity,
            "step_count": self.step_count,
            "reached_goal": terminated
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if self.state is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw mountain
        x = np.linspace(-1.2, 0.6, 100)
        y = np.sin(3 * x)
        ax.plot(x, y, 'b-', linewidth=2, label='Mountain')
        
        # Draw goal
        goal_y = np.sin(3 * self.goal_position)
        ax.plot(self.goal_position, goal_y, 'ro', markersize=10, label='Goal')
        
        # Draw car
        car_y = np.sin(3 * self.state[0])
        ax.plot(self.state[0], car_y, 'go', markersize=8, label='Car')
        
        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Position")
        ax.set_ylabel("Height")
        ax.set_title(f"MountainCar - Step {self.step_count}")
        ax.legend()
        ax.grid(True)
        
        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        
        plt.close(fig)
        return None
    
    def close(self):
        """Clean up environment resources."""
        pass


def create_default_gridworld() -> GridWorld:
    """Create a default GridWorld environment."""
    config = GridWorldConfig(
        width=8,
        height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=[(2, 2), (3, 2), (4, 2), (5, 2), (2, 5), (3, 5), (4, 5), (5, 5)]
    )
    return GridWorld(config)


def create_default_mountaincar() -> SimpleMountainCar:
    """Create a default MountainCar environment."""
    return SimpleMountainCar(goal_position=0.5, max_steps=200)
