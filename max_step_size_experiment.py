"""
Train multiple PPO agents on the festival environment with different configurations,
with multiple agents per configuration for averaging.
"""

import argparse
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from env import FestivalEnv, EnvConfig
from agents import PPOAgent
from simulation import FieldRenderer
from collections import defaultdict
import os

def train_multiple_ppo(
    env_configs,
    agents_per_config: int = 3,
    num_episodes: int = 1000,
    update_interval: int = 2000,
    render: bool = False,
    save_interval: int = 100,
    save_dir: str = "models"
):
    """Train multiple PPO agents with different environment configurations."""
    # Create directory for models
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environments and agents (3 agents per config)
    all_envs = []
    all_agents = []
    config_groups = []
    
    for config_idx, cfg in enumerate(env_configs):
        group_envs = []
        group_agents = []
        for agent_idx in range(agents_per_config):
            env = FestivalEnv(seed=42 + config_idx * 100 + agent_idx, cfg=cfg)
            agent = PPOAgent(
                action_space=env.action_space,
                img_channels=3,
                img_size=64,
                state_dim=4,
                hidden_dim=12,
                lr=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio=0.2,
                train_iters=10,
                batch_size=64,
            )
            group_envs.append(env)
            group_agents.append(agent)
        all_envs.extend(group_envs)
        all_agents.extend(group_agents)
        config_groups.append((cfg.max_steps, group_envs, group_agents))
    
    renderers = [FieldRenderer(env.cfg) if render else None for env in all_envs]
    
    # Training metrics (grouped by config then averaged)
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    running_rewards = [0] * len(all_agents)
    
    # Training loop
    total_steps = 0
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Initialize all environments
        observations = [env.reset()[0] for env in all_envs]
        #Reset trash in all environments
        for env in all_envs:
            env.create_trash()
        # Reset done flags
        dones = [False] * len(all_agents)
        
        # Episode-specific metrics
        episode_rewards = [0] * len(all_agents)
        episode_lengths = [0] * len(all_agents)
        episode_illegals = [0] * len(all_agents)
        episode_trash_picked = [0] * len(all_agents)
        episode_trash_left = [0] * len(all_agents)
        
        # Initialize renderers if needed
        if render:
            for i, renderer in enumerate(renderers):
                if renderer and i % agents_per_config == 0:  # Only render first agent of each group
                    renderer.reset_axes()
                    renderer.init_static(all_envs[i])
                    renderer.init_trash(all_envs[i])
        
        # Episode loop - alternate between all agents
        while not all(dones):
            for i, (agent, env, renderer) in enumerate(zip(all_agents, all_envs, renderers)):
                if dones[i]:
                    continue
                
                # Get action and value
                with torch.no_grad():
                    action, log_prob, value = agent.act(observations[i])
                
                # Take action
                next_obs, reward, done, trunc, info, num_trash_left = env.step(action)
                dones[i] = done or trunc
                
                # Store transition
                agent.store_transition(
                    obs=observations[i],
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=dones[i]
                )
                
                # Update metrics
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                if info.get("illegal", True):
                    episode_illegals[i] += 1
                if info.get("picked_up", True):
                    episode_trash_picked[i] += 1
                
                # Update observation
                observations[i] = next_obs
                total_steps += 1
                
                # Render environment if enabled (only first agent of each group)
                if render and renderer and i % agents_per_config == 0:
                    renderer.update(env, illegal=info.get("illegal", False))
                
                # Update policy if enough steps collected
                if len(agent.obs_buffer) >= update_interval:
                    agent.update()
                
                if dones[i]:
                    episode_trash_left[i] = num_trash_left
        
        # Organize metrics by config group
        for config_idx, (max_steps, _, _) in enumerate(config_groups):
            group_start = config_idx * agents_per_config
            group_end = group_start + agents_per_config
            
            # Store individual agent metrics
            for i in range(group_start, group_end):
                # Make sure we're appending to lists
                metrics[max_steps]['episode_rewards'][i - group_start].append(episode_rewards[i])
                metrics[max_steps]['episode_lengths'][i - group_start].append(episode_lengths[i])
                metrics[max_steps]['episode_illegals'][i - group_start].append(episode_illegals[i])
                metrics[max_steps]['episode_trash_picking_frequencies'][i - group_start].append(
                    episode_trash_picked[i] / episode_lengths[i] if episode_lengths[i] > 0 else 0
                )
                metrics[max_steps]['episode_trash_pieces_left'][i - group_start].append(episode_trash_left[i])
                running_rewards[i] = 0.95 * running_rewards[i] + 0.05 * episode_rewards[i]
        
        # Save models periodically
        if (episode + 1) % save_interval == 0:
            for config_idx, (max_steps, _, group_agents) in enumerate(config_groups):
                for agent_idx, agent in enumerate(group_agents):
                    save_path = os.path.join(save_dir, f"ppo_steps{max_steps}_run{agent_idx}.pt")
                    torch.save({
                        'episode': episode,
                        'model_state_dict': agent.actor_critic.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'running_reward': running_rewards[config_idx * agents_per_config + agent_idx],
                        'max_steps': max_steps
                    }, save_path)
            
            print(f"\nEpisode {episode + 1}")
            for max_steps, _, _ in config_groups:
                config_idx = next(i for i, cfg in enumerate(env_configs) if cfg.max_steps == max_steps)
                group_start = config_idx * agents_per_config
                group_end = group_start + agents_per_config
                
                avg_reward = np.mean(episode_rewards[group_start:group_end])
                avg_length = np.mean(episode_lengths[group_start:group_end])
                avg_illegal = np.mean(episode_illegals[group_start:group_end])
                avg_pick_freq = np.mean([episode_trash_picked[i]/episode_lengths[i] 
                                       for i in range(group_start, group_end) if episode_lengths[i] > 0])
                avg_trash_left = np.mean(episode_trash_left[group_start:group_end])
                
                print(f"\nConfig max_steps={max_steps} (averaged over {agents_per_config} runs)")
                print(f"Avg running reward: {np.mean(running_rewards[group_start:group_end]):.2f}")
                print(f"Avg episode reward: {avg_reward:.2f}")
                print(f"Avg episode length: {avg_length:.1f}")
                print(f"Avg illegal moves: {avg_illegal:.1f}")
                print(f"Avg trash picking frequency: {avg_pick_freq:.2f}")
                print(f"Avg trash pieces left: {avg_trash_left:.1f}")
            
            # Plot moving averages
            if episode >= 100:
                plot_moving_averages(metrics, window_size=100)
    
    return metrics

def plot_training_curves(metrics):
    """Plot training metrics averaged by configuration group."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    titles = [
        'Training Rewards',
        'Episode Lengths',
        'Illegal Moves per Episode',
        'Trash Picking Frequency per Episode',
        'Trash Pieces Left at Episode End'
    ]
    ylabels = ['Reward', 'Steps', 'Count', 'Frequency', 'Pieces Left']
    metric_keys = [
        'episode_rewards',
        'episode_lengths',
        'episode_illegals',
        'episode_trash_picking_frequencies',
        'episode_trash_pieces_left'
    ]
    
    # Get sorted list of max_steps values
    max_steps_list = sorted(metrics.keys())
    
    for ax, title, ylabel, key in zip(axes, titles, ylabels, metric_keys):
        for max_steps in max_steps_list:
            # Get all runs for this configuration
            runs_data = metrics[max_steps][key]
            
            # Skip if not a dictionary of lists
            if not isinstance(runs_data, dict):
                continue
                
            # Collect all valid runs
            valid_runs = []
            for run_idx, run_values in runs_data.items():
                if isinstance(run_values, list):
                    valid_runs.append(run_values)
            
            if not valid_runs:
                continue
                
            # Find the minimum length among all runs
            min_length = min(len(run) for run in valid_runs)
            truncated_runs = [run[:min_length] for run in valid_runs]
            runs_array = np.array(truncated_runs)
            avg_data = np.mean(runs_array, axis=0)
            
            ax.plot(avg_data, label=f'max_steps={max_steps}')
            
            # Show interquartile range if we have multiple runs
            if runs_array.shape[0] > 1:
                ax.fill_between(
                    range(min_length),
                    np.percentile(runs_array, 25, axis=0),
                    np.percentile(runs_array, 75, axis=0),
                    alpha=0.2
                )
        
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_averaged.png')
    plt.close()

def plot_moving_averages(metrics, window_size=100):
    """Plot moving averages of training metrics averaged by configuration group."""
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    titles = [
        'Reward Moving Average',
        'Illegal Moves Moving Average',
        'Trash Picking Frequency Moving Average',
        'Trash Pieces Left Moving Average'
    ]
    ylabels = ['Reward', 'Illegal Moves', 'Frequency', 'Pieces Left']
    metric_keys = [
        'episode_rewards',
        'episode_illegals',
        'episode_trash_picking_frequencies',
        'episode_trash_pieces_left'
    ]
    
    # Get sorted list of max_steps values
    max_steps_list = sorted(metrics.keys())
    
    for ax, title, ylabel, key in zip(axes, titles, ylabels, metric_keys):
        for max_steps in max_steps_list:
            # Get all runs for this configuration
            runs_data = metrics[max_steps][key]
            
            # Skip if not a dictionary of lists
            if not isinstance(runs_data, dict):
                continue
                
            # Collect all valid runs
            valid_runs = []
            for run_idx, run_values in runs_data.items():
                if isinstance(run_values, list):
                    valid_runs.append(run_values)
            
            if not valid_runs:
                continue
                
            # Calculate moving averages for each run
            ma_data = []
            for run in valid_runs:
                if len(run) >= window_size:
                    ma_data.append(moving_average(run, window_size))
            
            if not ma_data:
                continue
                
            # Find the minimum length among the moving averages
            min_length = min(len(ma) for ma in ma_data)
            truncated_ma = [ma[:min_length] for ma in ma_data]
            avg_ma = np.mean(truncated_ma, axis=0)
            
            ax.plot(avg_ma, label=f'max_steps={max_steps}')
            
            # Show interquartile range if we have multiple runs
            if len(truncated_ma) > 1:
                ax.fill_between(
                    range(len(avg_ma)),
                    np.percentile(truncated_ma, 25, axis=0),
                    np.percentile(truncated_ma, 75, axis=0),
                    alpha=0.2
                )
        
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('moving_averages_averaged.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000000)
    parser.add_argument("--agents-per-config", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--update-interval", type=int, default=1200)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--window-size", type=int, default=80)
    args = parser.parse_args()
    
    # Create different environment configurations
    env_configs = [
        EnvConfig(max_steps=100),
        EnvConfig(max_steps=400),
        EnvConfig(max_steps=1000),
        EnvConfig(max_steps=4000),
    ]
    
    # Train agents
    metrics = train_multiple_ppo(
        env_configs=env_configs,
        agents_per_config=args.agents_per_config,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        render=args.render,
        save_interval=args.save_interval,
        save_dir=args.save_dir
    )
    
    # Plot training curves
    plot_training_curves(metrics)
    plot_moving_averages(metrics, window_size=args.window_size)