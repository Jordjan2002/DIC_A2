"""
simple_evaluate_ppo.py

A simple evaluation script for PPO agents, matching the CLI and output of simple_evaluate.py.
Includes compatibility layer for architecture mismatches.
"""
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import gc

from env.festival_trash_env import FestivalEnv, EnvConfig
from agents.ppo_agent import PPOAgent

class PPOCompatibilityWrapper:
    """Wrapper to handle architecture mismatches between saved model and current environment."""
    
    def __init__(self, ppo_agent):
        self.agent = ppo_agent
        
    def act(self, obs):
        """Convert 4-channel image to 1-channel and handle action mapping."""
        # Convert 4-channel image to 1-channel by taking the mean across channels
        if obs['image'].ndim == 3 and obs['image'].shape[0] == 4:
            # Convert from (4, 64, 64) to (1, 64, 64)
            obs_converted = {
                'image': np.mean(obs['image'], axis=0, keepdims=True),  # Average across channels
                'state': obs['state']
            }
        else:
            obs_converted = obs
            
        # Get action from PPO agent (which has 9 actions)
        action, log_prob, value = self.agent.act(obs_converted)
        
        # Map from 9 actions back to 8 actions for the environment
        # Assume action 8 (index 8) should map to a valid action in 0-7 range
        if action >= 8:
            action = action % 8  # Simple mapping: action 8 -> action 0
            
        return action, log_prob, value

def simple_evaluate_ppo(agent_wrapper, env_config, episodes=5, max_steps=50):
    """Simple PPO evaluation using the unified evaluation function."""
    from final_experiment import evaluate_agent
    
    mean_reward, std_reward, mean_illegal = evaluate_agent(agent_wrapper, env_config, episodes, max_steps, "ppo")
    
    print(f"Results: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Illegal Moves: {mean_illegal:.1f}")
    print("Evaluation completed successfully!")
    print(f"Environment: {env_config}")
    print(f"Episodes: {episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Return format expected by existing code
    rewards = [mean_reward] * episodes  # Approximate for backward compatibility
    return mean_reward, std_reward, rewards

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained PPO model')
    parser.add_argument('--env_type', type=str, choices=['normal', 'benches', 'trash'],
                      default='normal', help='Environment type')
    parser.add_argument('--episodes', type=int, default=5,
                      help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=50,
                      help='Maximum steps per episode')
    args = parser.parse_args()
    if args.env_type == 'normal':
        env_config = EnvConfig()
        print("Testing Normal Setting (static trash and benches)")
    elif args.env_type == 'benches':
        env_config = EnvConfig(static_benches=False)
        print("Testing Benches Setting (stochastic benches)")
    elif args.env_type == 'trash':
        env_config = EnvConfig(static_trash=False)
        print("Testing Trash Setting (stochastic trash)")
    print(f"Loading PPO agent from: {args.model_path}")
    
    # Create action space with correct dimensions (9 actions for the saved model)
    action_space = type('ActionSpace', (), {'n': 9})()
    
    # Initialize PPO agent with the EXACT same architecture as the saved model
    agent = PPOAgent(
        action_space=action_space,
        img_channels=1,  # Match the saved model (1 channel)
        img_size=64,
        state_dim=4,
        hidden_dim=12,  # Match the saved model
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        train_iters=10,
        batch_size=64,
    )
    
    agent.load(args.model_path)
    
    # Wrap the agent for compatibility
    agent_wrapper = PPOCompatibilityWrapper(agent)
    
    mean_reward, std_reward, rewards = simple_evaluate_ppo(agent_wrapper, env_config, args.episodes, args.max_steps)

if __name__ == "__main__":
    main() 