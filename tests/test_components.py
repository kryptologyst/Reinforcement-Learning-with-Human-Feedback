"""
Unit Tests: Test suite for RL components.

This module provides comprehensive unit tests for the RL environment,
agents, and utilities.
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.envs.rlhf_env import RLHFEnvironment, RLHFConfig, create_default_rlhf_env
from src.envs.mock_envs import GridWorld, GridWorldConfig, SimpleMountainCar, create_default_gridworld
from src.agents.modern_agents import PPOAgent, DQNAgent, RainbowDQNAgent, AgentConfig
from src.utils.training_utils import TrainingConfig, Logger, CheckpointManager


class TestRLHFEnvironment(unittest.TestCase):
    """Test cases for RLHF environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RLHFConfig(
            prompts=["Test prompt"],
            responses={"Test prompt": ["Response 1", "Response 2", "Response 3"]},
            preferred_responses={"Test prompt": "Response 1"}
        )
        self.env = RLHFEnvironment(self.config)
    
    def test_environment_creation(self):
        """Test environment creation."""
        self.assertIsInstance(self.env, RLHFEnvironment)
        self.assertEqual(self.env.action_space.n, 3)
        self.assertEqual(self.env.observation_space.shape, (1,))
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIn("prompt", info)
        self.assertEqual(self.env.step_count, 0)
    
    def test_step(self):
        """Test environment step."""
        obs, _ = self.env.reset()
        action = 0
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsInstance(next_obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("prompt", info)
        self.assertIn("response", info)
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        obs, _ = self.env.reset()
        
        # Test preferred response
        action = 0  # Should be preferred response
        _, reward, _, _, _ = self.env.step(action)
        self.assertEqual(reward, 1.0)
        
        # Test non-preferred response
        obs, _ = self.env.reset()
        action = 1  # Should not be preferred response
        _, reward, _, _, _ = self.env.step(action)
        self.assertEqual(reward, 0.0)


class TestGridWorldEnvironment(unittest.TestCase):
    """Test cases for GridWorld environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GridWorldConfig(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 3))
        self.env = GridWorld(self.config)
    
    def test_environment_creation(self):
        """Test environment creation."""
        self.assertIsInstance(self.env, GridWorld)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.observation_space.shape, (2,))
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs[0], 0)  # Start position x
        self.assertEqual(obs[1], 0)  # Start position y
        self.assertIn("position", info)
    
    def test_step(self):
        """Test environment step."""
        obs, _ = self.env.reset()
        action = 3  # Right
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsInstance(next_obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("position", info)
    
    def test_boundary_conditions(self):
        """Test boundary conditions."""
        obs, _ = self.env.reset()
        
        # Try to move left from start position (should stay in place)
        action = 2  # Left
        next_obs, reward, _, _, _ = self.env.step(action)
        self.assertEqual(next_obs[0], 0)  # Should stay at x=0
        self.assertEqual(next_obs[1], 0)  # Should stay at y=0


class TestPPOAgent(unittest.TestCase):
    """Test cases for PPO agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AgentConfig(learning_rate=0.001, device="cpu")
        self.agent = PPOAgent(state_dim=4, action_dim=2, config=self.config)
    
    def test_agent_creation(self):
        """Test agent creation."""
        self.assertIsInstance(self.agent, PPOAgent)
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.value_net)
    
    def test_get_action(self):
        """Test action selection."""
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action, log_prob = self.agent.get_action(state, training=True)
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)
        self.assertIn(action, [0, 1])
    
    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action, log_prob = self.agent.get_action(state, training=False)
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)
        self.assertIn(action, [0, 1])


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQN agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AgentConfig(learning_rate=0.001, device="cpu")
        self.agent = DQNAgent(state_dim=4, action_dim=2, config=self.config)
    
    def test_agent_creation(self):
        """Test agent creation."""
        self.assertIsInstance(self.agent, DQNAgent)
        self.assertIsNotNone(self.agent.q_net)
        self.assertIsNotNone(self.agent.target_net)
    
    def test_get_action(self):
        """Test action selection."""
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action = self.agent.get_action(state, training=True)
        
        self.assertIsInstance(action, int)
        self.assertIn(action, [0, 1])
    
    def test_store_transition(self):
        """Test transition storage."""
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action = 0
        reward = 1.0
        next_state = np.array([0.0, 1.0, 0.0, 0.0])
        done = False
        
        self.agent.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.buffer), 1)
    
    def test_update(self):
        """Test agent update."""
        # Add some transitions to buffer
        for _ in range(100):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.choice([True, False])
            self.agent.store_transition(state, action, reward, next_state, done)
        
        # Test update
        initial_loss = None
        self.agent.update()
        # Update should complete without error


class TestRainbowDQNAgent(unittest.TestCase):
    """Test cases for Rainbow DQN agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AgentConfig(learning_rate=0.001, device="cpu")
        self.agent = RainbowDQNAgent(state_dim=4, action_dim=2, config=self.config)
    
    def test_agent_creation(self):
        """Test agent creation."""
        self.assertIsInstance(self.agent, RainbowDQNAgent)
        self.assertIsNotNone(self.agent.q_net)
        self.assertIsNotNone(self.agent.target_net)
    
    def test_get_action(self):
        """Test action selection."""
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action = self.agent.get_action(state, training=True)
        
        self.assertIsInstance(action, int)
        self.assertIn(action, [0, 1])


class TestTrainingUtils(unittest.TestCase):
    """Test cases for training utilities."""
    
    def test_training_config_creation(self):
        """Test training configuration creation."""
        config = TrainingConfig()
        self.assertIsInstance(config, TrainingConfig)
        self.assertEqual(config.env_name, "rlhf")
        self.assertEqual(config.agent_type, "ppo")
    
    def test_checkpoint_manager(self):
        """Test checkpoint manager."""
        manager = CheckpointManager("test_checkpoints")
        self.assertIsInstance(manager, CheckpointManager)
        
        # Test checkpoint directory creation
        self.assertTrue(manager.checkpoint_dir.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_rlhf_training_integration(self):
        """Test RLHF training integration."""
        env = create_default_rlhf_env()
        config = AgentConfig(device="cpu")
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config
        )
        
        # Run a few episodes
        for _ in range(5):
            obs, _ = env.reset()
            done = False
            while not done:
                action, log_prob = agent.get_action(obs, training=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        
        env.close()
    
    def test_gridworld_training_integration(self):
        """Test GridWorld training integration."""
        env = create_default_gridworld()
        config = AgentConfig(device="cpu")
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config
        )
        
        # Run a few episodes
        for _ in range(5):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs, training=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(obs, action, reward, obs, done)
                done = terminated or truncated
            
            # Update agent
            agent.update()
        
        env.close()


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestRLHFEnvironment,
        TestGridWorldEnvironment,
        TestPPOAgent,
        TestDQNAgent,
        TestRainbowDQNAgent,
        TestTrainingUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
