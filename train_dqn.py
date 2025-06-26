"""
train_dqn.py
Author: Jord & Stijn

A modular implementation of DQN training for the festival environment.
Supports command line arguments for easy parameter configuration.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch

from env.simple_festival_env import SimpleFestivalEnv
from env.festival_trash_env import FestivalEnv
from agents.dqn_agent import DQNAgent
from simulation import FieldRenderer


# visualization parameters
VIZ_INTERVAL = 500     # run a GUI episode every 
MOVING_AVG_WINDOW = 50 # size of moving‑average window for the plot


@dataclass
class TrainingConfig:
    """Configuration class for DQN training parameters."""
    episodes: int = 500
    max_steps: int = 100
    alpha: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.998
    memory_size: int = 10000
    batch_size: int = 64
    target_update: int = 100
    random_seed: int = 42
    show_gui: bool = False
    save_results: bool = True
    warmup_steps: int = 20000
    print_every: int = 50

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create a TrainingConfig instance from parsed command line arguments."""
        return cls(
            episodes=args.episodes,
            max_steps=args.max_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
            random_seed=args.random_seed,
            show_gui=args.show_gui,
            save_results=args.save_results,
            warmup_steps=args.warmup_steps,
            print_every=args.print_every
        )


def run_gui_episode(agent: DQNAgent,
                    max_steps: int) -> None:
    """Play one full episode with the GUI so you can watch the policy."""
    env = FestivalEnv()
    renderer = FieldRenderer(env.cfg)
    
    state, _ = env.reset()
    renderer.reset_axes()
    renderer.init_static(env)
    renderer.init_trash(env)
    renderer.update(env)
    plt.ion()  # Turn on interactive mode
    
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        renderer.update(env)
        plt.draw()
        plt.pause(0.01)  # Use pause instead of sleep

    print(f"GUI Episode finished with reward: {total_reward}")
    plt.pause(1)  # Pause to show final state
    plt.close('all')  # Close the plot window
    plt.ioff()  # Turn off interactive mode


def train_agent(
    episodes: int = 1000,
    max_steps: int = 100,
    alpha: float = 0.001,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay: float = 0.998,
    memory_size: int = 100000,
    batch_size: int = 128,
    target_update: int = 10,
    random_seed: int | None = None,
    show_gui: bool = False,
    save_results: bool = True,
    warmup_steps: int = 20000,
    print_every: int = 50,
    viz_interval: int = 10
) -> tuple[DQNAgent, list[float]]:
    """Train a DQN agent on the festival environment."""
    # Set up random seeds for reproducibility
    if random_seed is not None:
        # Generate a random seed if none provided
        random_seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {random_seed}")
    
    # Set random seeds for all sources of randomness
    random.seed(random_seed)  # Python random
    np.random.seed(random_seed)  # NumPy random
    torch.manual_seed(random_seed)  # PyTorch random
    if torch.cuda.is_available():  # GPU random
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # Multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment seed
    train_env = FestivalEnv(seed=random_seed)
    state, _ = train_env.reset()
    
    print(f"Environment input shape: {state['image'].shape}")
    
    # Initialize DQN agent (no need to set random seeds in agent as they're already set)
    agent = DQNAgent(
        n_actions=8,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update=target_update,
        normalize_rewards=True,  # Enable reward normalization
        seed=None  # Seeds already set above
    )

    # Warm-up phase: fill replay buffer with random actions
    print("Warming up replay buffer...")
    warmup = True
    state, _ = train_env.reset()
    for _ in tqdm(range(warmup_steps), desc="Warm-up"):
        action = random.randrange(agent.n_actions)  # Random action
        next_state, reward, done, _, info = train_env.step(action)
        agent.update(state, action, reward, next_state, done, warmup)
        state = next_state if not done else train_env.reset()[0]
    print("Warm-up complete!")
    print(f"Initial reward norm: mean={agent.reward_mean:.3f}, std={agent.reward_std:.3f}")

    episode_returns: list[float] = []

    # training loop
    for ep in range(1, episodes + 1):
        state, _ = train_env.reset()
        done = False
        G = 0
        step = 0
        bins_used = 0
        pickups = 0

        # Create progress bar for current episode
        pbar = tqdm(total=max_steps, desc=f"Episode {ep}", leave=False)
        
        while not done and step < max_steps:
            action = agent.act(state)
            next_state, reward, done, _, info = train_env.step(action)
            
            # Track bin usage and pickups for logging
            if 5.0 <= reward <= 15.0:  # pickup rewards
                pickups += 1
            elif reward > 15.0:  # unload rewards
                bins_used += 1
                
            step += 1
            G += reward

            agent.update(state, action, reward, next_state, done, False)
            state = next_state

            pbar.update(1)
            pbar.set_postfix({"reward": f"{G:.1f}"})

        pbar.close()
        agent.end_episode()
        episode_returns.append(G)

        # Calculate moving average
        window_size = min(10, len(episode_returns))
        ma = np.mean(episode_returns[-window_size:])
        
        # Print episode summary with bin and pickup info
        print(f"Episode {ep:>4} | Return {G:>6.1f} | MovingAvg {ma:>6.1f} | ε {agent.epsilon:5.3f} | Pickups: {pickups} | Bins: {bins_used}")

        # Print reward normalization stats periodically
        if ep % print_every == 0:
            print(f"  Reward norm: mean={agent.reward_mean:.3f}, std={agent.reward_std:.3f}, count={agent.reward_count}")

        # show with GUI
        if show_gui and ep % viz_interval == 0: 
            print(f"\n Visualising policy after episode {ep} …")
            run_gui_episode(agent, max_steps)
            print("Training continues …\n")

    if save_results:
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        Path("results").mkdir(exist_ok=True)
        agent.save(Path("results") / f"dqn_model_{timestamp}.pt")
        print(f"\nTraining finished – model saved as results/dqn_model_{timestamp}.pt")

    return agent, episode_returns


def plot_learning_curve(returns: list[float], episodes: int, window_size: int = 100):
    """Plot the learning curve with moving average."""
    plt.figure(figsize=(10, 5))
    plt.plot(returns, alpha=0.3, label="episode return")
    if len(returns) >= window_size:
        ma = np.convolve(
            returns,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        plt.plot(
            range(window_size - 1, len(returns)),
            ma,
            label=f"{window_size}-episode moving average"
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics(agent: DQNAgent, returns: list[float]):
    """Plot training metrics including loss, Q-values, and returns."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot returns
    ax1.plot(returns, label='Episode Return')
    ax1.set_title('Episode Returns')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(agent.losses, label='Training Loss')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Plot Q-values
    ax3.plot(agent.q_values, label='Mean Q-value')
    ax3.set_title('Mean Q-values')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Q-value')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a DQN agent on the festival environment.')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=500,
                      help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                      help='Maximum steps per episode')
    parser.add_argument('--alpha', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0,
                      help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.05,
                      help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.998,
                      help='Exploration rate decay')
    parser.add_argument('--memory-size', type=int, default=10000,
                      help='Size of replay memory')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--target-update', type=int, default=100,
                      help='Target network update frequency')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--warmup-steps', type=int, default=20000,
                      help='Number of random steps for replay buffer warmup')
    parser.add_argument('--print-every', type=int, default=50,
                      help='Print stats every N episodes')
    
    # Flags
    parser.add_argument('--show-gui', action='store_true',
                      help='Show GUI visualization during training')
    parser.add_argument('--no-save', dest='save_results', action='store_false',
                      help='Disable saving of results')
    parser.set_defaults(save_results=True)
    
    return parser.parse_args()

def main():
    """Main training loop with command line argument support."""
    args = parse_args()
    config = TrainingConfig.from_args(args)
    
    # Train the agent
    agent, returns = train_agent(
        episodes=config.episodes,
        max_steps=config.max_steps,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update=config.target_update,
        random_seed=config.random_seed,
        show_gui=config.show_gui,
        save_results=config.save_results
    )
    
    # Plot results
    plot_learning_curve(returns, config.episodes)
    plot_metrics(agent, returns)

if __name__ == "__main__":
    main() 