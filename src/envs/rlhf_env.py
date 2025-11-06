"""
RLHF Environment: Simulated Human Feedback for Reinforcement Learning

This module implements a simulated RLHF environment where an agent learns to generate
responses to prompts based on human preference feedback.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class RLHFConfig:
    """Configuration for RLHF environment."""
    prompts: List[str]
    responses: Dict[str, List[str]]
    preferred_responses: Dict[str, str]
    max_episode_steps: int = 100
    reward_scale: float = 1.0


class RLHFEnvironment(gym.Env):
    """
    RLHF Environment for training agents with human feedback.
    
    The agent receives prompts and generates responses, receiving rewards
    based on simulated human preferences.
    """
    
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.prompts = config.prompts
        self.responses = config.responses
        self.preferred_responses = config.preferred_responses
        
        # Action space: index of response to choose
        self.action_space = spaces.Discrete(max(len(responses) for responses in self.responses.values()))
        
        # Observation space: one-hot encoded prompt
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.prompts),), dtype=np.float32
        )
        
        self.current_prompt_idx: Optional[int] = None
        self.step_count = 0
        self.max_steps = config.max_episode_steps
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_prompt_idx = self.np_random.integers(0, len(self.prompts))
        self.step_count = 0
        
        observation = self._get_observation()
        info = {"prompt": self.prompts[self.current_prompt_idx]}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_prompt_idx is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        prompt = self.prompts[self.current_prompt_idx]
        available_responses = self.responses[prompt]
        
        # Ensure action is valid
        action = min(action, len(available_responses) - 1)
        response = available_responses[action]
        
        # Calculate reward based on human preference
        reward = self._calculate_reward(prompt, response)
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Get next observation (new random prompt)
        observation = self._get_observation()
        
        info = {
            "prompt": prompt,
            "response": response,
            "action": action,
            "step_count": self.step_count,
            "preferred_response": self.preferred_responses.get(prompt, "")
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (one-hot encoded prompt)."""
        obs = np.zeros(len(self.prompts), dtype=np.float32)
        if self.current_prompt_idx is not None:
            obs[self.current_prompt_idx] = 1.0
        return obs
    
    def _calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward based on human preference."""
        preferred = self.preferred_responses.get(prompt)
        if preferred and response == preferred:
            return self.config.reward_scale
        return 0.0
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if self.current_prompt_idx is None:
            return "Environment not initialized"
        
        prompt = self.prompts[self.current_prompt_idx]
        return f"Prompt: {prompt}\nStep: {self.step_count}/{self.max_steps}"
    
    def close(self):
        """Clean up environment resources."""
        pass


def create_default_rlhf_env() -> RLHFEnvironment:
    """Create a default RLHF environment with sample prompts and responses."""
    config = RLHFConfig(
        prompts=["Tell me a joke", "Say something wise", "Motivate me"],
        responses={
            "Tell me a joke": ["Why did the chicken cross the road?", "I don't know any jokes.", "Knock knock!"],
            "Say something wise": ["Knowledge is power.", "Life is short.", "YOLO."],
            "Motivate me": ["You got this!", "Give up now.", "Keep pushing forward!"]
        },
        preferred_responses={
            "Tell me a joke": "Knock knock!",
            "Say something wise": "Knowledge is power.",
            "Motivate me": "Keep pushing forward!"
        }
    )
    return RLHFEnvironment(config)
