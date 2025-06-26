"""
simple_evaluate.py

A simple evaluation script that loads a trained DQN agent and evaluates it
on different environment configurations without the complex safety measures.
"""
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import gc

from env.festival_trash_env import EnvConfig
from agents.dqn_agent import DQNAgent
from final_experiment import evaluate_agent

def simple_evaluate(agent, env_config, episodes=10, max_steps=100):
    """Simple evaluation using the unified evaluation function."""
    mean_reward, std_reward, mean_illegal = evaluate_agent(agent, env_config, episodes, max_steps, "dqn")
    
    print(f"\nResults: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Illegal Moves: {mean_illegal:.1f}")
    
    return mean_reward, std_reward

def main():
    parser = argparse.ArgumentParser(description='Simple evaluation of trained DQN agent')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--env_type', type=str, choices=['normal', 'benches', 'trash'], 
                       default='normal', help='Environment type to test')
    args = parser.parse_args()
    
    print(f"Loading agent from: {args.model_path}")
    
    try:
        # Load the agent
        agent = DQNAgent()
        agent.load(Path(args.model_path))
        print("Agent loaded successfully.")
        
        # Set up environment configuration
        config = EnvConfig()
        if args.env_type == 'normal':
            config.static_trash = True
            config.static_benches = True
            print("Testing Normal Setting (static trash and benches)")
        elif args.env_type == 'benches':
            config.static_benches = False
            config.static_trash = True
            print("Testing Stochastic Benches Setting")
        elif args.env_type == 'trash':
            config.static_trash = False
            config.static_benches = True
            print("Testing Stochastic Trash Setting")
        
        # Run evaluation
        mean_reward, std_reward = simple_evaluate(agent, config, args.episodes, args.max_steps)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Environment: {args.env_type}")
        print(f"Episodes: {args.episodes}")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 