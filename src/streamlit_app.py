"""
Streamlit UI: Interactive interface for RL training and visualization.

This module provides a web-based interface for training RL agents,
monitoring progress, and visualizing results.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
from pathlib import Path
import sys
import time
import threading
import queue

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.envs.rlhf_env import create_default_rlhf_env
from src.envs.mock_envs import create_default_gridworld, create_default_mountaincar
from src.agents.modern_agents import PPOAgent, DQNAgent, RainbowDQNAgent, AgentConfig
from src.utils.training_utils import TrainingConfig, Logger, CheckpointManager


def create_environment(env_name: str):
    """Create environment based on name."""
    if env_name == "RLHF":
        return create_default_rlhf_env()
    elif env_name == "GridWorld":
        return create_default_gridworld()
    elif env_name == "MountainCar":
        return create_default_mountaincar()
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def create_agent(env, agent_type: str, config: dict):
    """Create agent based on type and configuration."""
    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.n
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    
    agent_config = AgentConfig(**config)
    
    if agent_type == "PPO":
        return PPOAgent(state_dim, action_dim, agent_config)
    elif agent_type == "DQN":
        return DQNAgent(state_dim, action_dim, agent_config)
    elif agent_type == "Rainbow DQN":
        return RainbowDQNAgent(state_dim, action_dim, agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_training_session(env_name: str, agent_type: str, config: dict, 
                        progress_queue: queue.Queue, stop_event: threading.Event):
    """Run training session in background thread."""
    try:
        # Create environment and agent
        env = create_environment(env_name)
        agent = create_agent(env, agent_type, config)
        
        episode = 0
        episode_rewards = []
        episode_lengths = []
        
        while not stop_event.is_set() and episode < config.get('max_episodes', 1000):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and not stop_event.is_set():
                if agent_type == "PPO":
                    action, log_prob = agent.get_action(obs, training=True)
                else:
                    action = agent.get_action(obs, training=True)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Store transition for DQN agents
                if agent_type in ["DQN", "Rainbow DQN"]:
                    agent.store_transition(obs, action, reward, obs, done)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Update agent
            if agent_type == "PPO":
                # Simple PPO update (in practice, you'd collect more data)
                pass
            elif agent_type in ["DQN", "Rainbow DQN"]:
                agent.update()
            
            # Send progress update
            progress_queue.put({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
                'avg_length': np.mean(episode_lengths[-100:]) if episode_lengths else 0
            })
            
            episode += 1
            
            # Small delay to prevent overwhelming the UI
            time.sleep(0.01)
        
        progress_queue.put({'status': 'completed'})
        
    except Exception as e:
        progress_queue.put({'status': 'error', 'error': str(e)})
    finally:
        env.close()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Training Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning Training Dashboard")
    st.markdown("Interactive interface for training and monitoring RL agents")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Environment selection
        env_name = st.selectbox(
            "Environment",
            ["RLHF", "GridWorld", "MountainCar"],
            help="Choose the environment to train on"
        )
        
        # Agent selection
        agent_type = st.selectbox(
            "Agent Type",
            ["PPO", "DQN", "Rainbow DQN"],
            help="Choose the RL algorithm"
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        max_episodes = st.number_input("Max Episodes", min_value=100, max_value=10000, value=1000)
        learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=3e-4, format="%.2e")
        gamma = st.number_input("Gamma (Discount Factor)", min_value=0.1, max_value=0.99, value=0.99, step=0.01)
        epsilon = st.number_input("Epsilon (for DQN)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        
        # Device selection
        device = st.selectbox("Device", ["cpu", "cuda"], help="Choose the device for training")
        
        # Training configuration
        config = {
            'max_episodes': max_episodes,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon': epsilon,
            'device': device,
            'batch_size': 64,
            'buffer_size': 10000
        }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Progress")
        
        # Initialize session state
        if 'training_data' not in st.session_state:
            st.session_state.training_data = []
        if 'training_active' not in st.session_state:
            st.session_state.training_active = False
        if 'progress_queue' not in st.session_state:
            st.session_state.progress_queue = queue.Queue()
        if 'stop_event' not in st.session_state:
            st.session_state.stop_event = threading.Event()
        
        # Training controls
        col_start, col_stop, col_clear = st.columns([1, 1, 1])
        
        with col_start:
            if st.button("🚀 Start Training", disabled=st.session_state.training_active):
                st.session_state.training_active = True
                st.session_state.training_data = []
                st.session_state.stop_event.clear()
                st.session_state.progress_queue = queue.Queue()
                
                # Start training thread
                training_thread = threading.Thread(
                    target=run_training_session,
                    args=(env_name, agent_type, config, 
                          st.session_state.progress_queue, st.session_state.stop_event)
                )
                training_thread.daemon = True
                training_thread.start()
                
                st.success("Training started!")
        
        with col_stop:
            if st.button("⏹️ Stop Training", disabled=not st.session_state.training_active):
                st.session_state.stop_event.set()
                st.session_state.training_active = False
                st.warning("Training stopped!")
        
        with col_clear:
            if st.button("🗑️ Clear Data"):
                st.session_state.training_data = []
                st.rerun()
        
        # Progress display
        if st.session_state.training_active:
            # Check for progress updates
            try:
                while True:
                    update = st.session_state.progress_queue.get_nowait()
                    
                    if update['status'] == 'completed':
                        st.session_state.training_active = False
                        st.success("Training completed!")
                        break
                    elif update['status'] == 'error':
                        st.session_state.training_active = False
                        st.error(f"Training error: {update['error']}")
                        break
                    else:
                        st.session_state.training_data.append(update)
                        
            except queue.Empty:
                pass
            
            # Auto-refresh every second
            time.sleep(1)
            st.rerun()
        
        # Plot training progress
        if st.session_state.training_data:
            df = pd.DataFrame(st.session_state.training_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Episode Rewards", "Episode Lengths", 
                              "Average Rewards (100 episodes)", "Average Lengths (100 episodes)"),
                vertical_spacing=0.1
            )
            
            # Episode rewards
            fig.add_trace(
                go.Scatter(x=df['episode'], y=df['reward'], 
                          mode='lines', name='Reward', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Episode lengths
            fig.add_trace(
                go.Scatter(x=df['episode'], y=df['length'], 
                          mode='lines', name='Length', line=dict(color='green')),
                row=1, col=2
            )
            
            # Average rewards
            fig.add_trace(
                go.Scatter(x=df['episode'], y=df['avg_reward'], 
                          mode='lines', name='Avg Reward', line=dict(color='red')),
                row=2, col=1
            )
            
            # Average lengths
            fig.add_trace(
                go.Scatter(x=df['episode'], y=df['avg_length'], 
                          mode='lines', name='Avg Length', line=dict(color='orange')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current statistics
            if len(df) > 0:
                latest = df.iloc[-1]
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Current Episode", latest['episode'])
                
                with col_stat2:
                    st.metric("Current Reward", f"{latest['reward']:.2f}")
                
                with col_stat3:
                    st.metric("Avg Reward (100)", f"{latest['avg_reward']:.2f}")
                
                with col_stat4:
                    st.metric("Avg Length (100)", f"{latest['avg_length']:.2f}")
    
    with col2:
        st.header("Environment Info")
        
        # Display environment information
        env = create_environment(env_name)
        
        st.subheader(f"{env_name} Environment")
        st.write(f"**Observation Space:** {env.observation_space}")
        st.write(f"**Action Space:** {env.action_space}")
        
        if hasattr(env, 'config'):
            st.write("**Configuration:**")
            st.json(env.config.__dict__)
        
        # Render environment
        st.subheader("Environment Visualization")
        if st.button("Render Environment"):
            try:
                obs, _ = env.reset()
                rendered = env.render(mode="rgb_array")
                if rendered is not None:
                    st.image(rendered, caption="Environment State")
                else:
                    st.info("Environment rendering not available")
            except Exception as e:
                st.error(f"Rendering error: {e}")
        
        # Training statistics
        st.subheader("Training Statistics")
        if st.session_state.training_data:
            df = pd.DataFrame(st.session_state.training_data)
            
            st.write("**Recent Performance:**")
            recent_df = df.tail(10)
            st.dataframe(recent_df[['episode', 'reward', 'length', 'avg_reward']])
            
            # Performance metrics
            if len(df) > 0:
                st.write("**Overall Performance:**")
                st.write(f"Total Episodes: {len(df)}")
                st.write(f"Best Reward: {df['reward'].max():.2f}")
                st.write(f"Average Reward: {df['reward'].mean():.2f}")
                st.write(f"Latest Avg Reward: {df['avg_reward'].iloc[-1]:.2f}")
        
        env.close()
    
    # Footer
    st.markdown("---")
    st.markdown("**RL Training Dashboard** - Built with Streamlit and PyTorch")


if __name__ == "__main__":
    main()
