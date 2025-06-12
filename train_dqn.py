"""
train_dqn.py
Author: Jord & Stijn

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

from env.simple_festival_env import SimpleFestivalEnv
from agents.dqn_agent import DQNAgent
from simulation import FieldRenderer


# visualization parameters
VIZ_INTERVAL = 500     # run a GUI episode every 
MOVING_AVG_WINDOW = 50 # size of moving‑average window for the plot


def run_gui_episode(agent: DQNAgent,
                    max_steps: int) -> None:
    """Play one full episode with the GUI so you can watch the policy."""
    env = SimpleFestivalEnv()
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
    random_seed: int = 2025,
    show_gui: bool = False,
    save_results: bool = True
) -> tuple[DQNAgent, list[float]]:
    # setting up the environment
    train_env = SimpleFestivalEnv(seed=random_seed)
    state, _ = train_env.reset()
    
    print(f"Environment input shape: {state['image'].shape}")
    
    # Initialize DQN agent
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
        seed=random_seed
    )

    # Warm-up phase: fill replay buffer with random actions
    print("Warming up replay buffer...")
    warmup = True
    warmup_steps = 20000 
    state, _ = train_env.reset()
    for _ in tqdm(range(warmup_steps), desc="Warm-up"):
        action = random.randrange(agent.n_actions)  # Random action
        next_state, reward, done, _, _ = train_env.step(action)
        agent.update(state, action, reward, next_state, warmup)
        state = next_state
        if done:
            state, _ = train_env.reset()
    print("Warm-up complete!")
    print(f"Initial reward norm: mean={agent.reward_mean:.3f}, std={agent.reward_std:.3f}")

    episode_returns: list[float] = []
    # best_return = float('-inf')
    # no_improvement_count = 0

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

            agent.update(state, action, reward, next_state)
            state = next_state

            pbar.update(1)
            pbar.set_postfix({"reward": f"{G:.1f}"})

        pbar.close()
        agent.end_episode()
        episode_returns.append(G)

        # Early stopping if performance plateaus
        # if G > best_return:
        #     best_return = G
        #     no_improvement_count = 0
        # else:
        #     no_improvement_count += 1

        # Calculate moving average
        ma = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else np.nan
        
        # Print episode summary with bin and pickup info
        print(f"Episode {ep:>4} | Return {G:>6.1f} | MovingAvg {ma:>6.1f} | ε {agent.epsilon:5.3f} | Pickups: {pickups} | Bins: {bins_used}")

        # Print reward normalization stats every 50 episodes
        if ep % 50 == 0:
            print(f"  Reward norm: mean={agent.reward_mean:.3f}, std={agent.reward_std:.3f}, count={agent.reward_count}")

        # show with GUI
        if show_gui and ep % 10 == 0: 
            print(f"\n Visualising policy after episode {ep} …")
            run_gui_episode(agent, max_steps)
            print("Training continues …\n")

        # Early stopping
        # if no_improvement_count >= 100: 
        #     print(f"\nEarly stopping at episode {ep} - no improvement for 100 episodes")
        #     break

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


if __name__ == "__main__":
    # Training parameters
    episodes = 500
    alpha = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05      # Lower min for more exploitation
    epsilon_decay = 0.998   # Faster decay
    memory_size = 10000
    batch_size = 64         # Smaller batch for more frequent updates
    target_update = 100     # More frequent target updates
    random_seed = 42
    print_every = 50
    
    agent, returns = train_agent(
        episodes=episodes,
        max_steps=100,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update=target_update,
        random_seed=random_seed,
        show_gui=False
    )
    
    plot_learning_curve(returns, episodes)
    plot_metrics(agent, returns) 