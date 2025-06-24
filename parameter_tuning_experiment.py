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
from itertools import product

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
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 30))
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

def train_multiple_ppo(
    gamma_and_clip_values,
    env_config,
    agents_per_config: int = 3,
    num_episodes: int = 1000,
    update_interval: int = 2000,
    render: bool = False,
    save_interval: int = 100,
    save_dir: str = "models"
):
    """Train multiple PPO agents with different PPO hyperparameters."""
    os.makedirs(save_dir, exist_ok=True)

    # Prepare groups
    config_groups = []  # list of tuples (gamma, clip, envs, agents)
    for gamma, clip in gamma_and_clip_values:
        group_envs = [FestivalEnv(seed=42 + i, cfg=env_config) for i in range(agents_per_config)]
        group_agents = [PPOAgent(
            action_space=env.action_space,
            img_channels=1,
            img_size=64,
            state_dim=4,
            hidden_dim=12,
            lr=3e-4,
            gamma=gamma,
            gae_lambda=0.95,
            clip_ratio=clip,
            train_iters=10,
            batch_size=64,
        ) for env in group_envs]
        config_groups.append((gamma, clip, group_envs, group_agents))

    # Flatten lists for parallel loops
    all_envs = [env for _, _, envs, _ in config_groups for env in envs]
    all_agents = [agent for _, _, _, agents in config_groups for agent in agents]
    renderers = [FieldRenderer(env.cfg) if render else None for env in all_envs]

    # Metrics storage: metrics[(gamma, clip)][metric_name][run_idx] = list of values per episode
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    running_rewards = [0.0] * len(all_agents)

    total_steps = 0
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset all envs
        observations = [env.reset()[0] for env in all_envs]
        for env in all_envs:
            env.create_trash()
        dones = [False] * len(all_agents)

        # Episode-level accumulators
        ep_rewards = [0.0] * len(all_agents)
        ep_lengths = [0] * len(all_agents)
        ep_illegals = [0] * len(all_agents)
        ep_picked = [0] * len(all_agents)
        ep_left = [0] * len(all_agents)

        # Init renderers
        if render:
            for i, ren in enumerate(renderers):
                if ren and i % agents_per_config == 0:
                    ren.reset_axes()
                    ren.init_static(all_envs[i])
                    ren.init_trash(all_envs[i])

        # Step through until all done
        while not all(dones):
            for idx, (agent, env, ren) in enumerate(zip(all_agents, all_envs, renderers)):
                if dones[idx]:
                    continue
                with torch.no_grad():
                    action, logp, val = agent.act(observations[idx])
                next_obs, reward, done, trunc, info, trash_left = env.step(action)
                dones[idx] = done or trunc

                agent.store_transition(
                    obs=observations[idx],
                    action=action,
                    reward=reward,
                    value=val,
                    log_prob=logp,
                    done=dones[idx]
                )

                ep_rewards[idx] += reward
                ep_lengths[idx] += 1
                if info.get("illegal", False):
                    ep_illegals[idx] += 1
                if info.get("picked_up", False):
                    ep_picked[idx] += 1

                observations[idx] = next_obs
                total_steps += 1

                if render and ren and idx % agents_per_config == 0:
                    ren.update(env, illegal=info.get("illegal", False))

                if len(agent.obs_buffer) >= update_interval:
                    agent.update()

                if dones[idx]:
                    ep_left[idx] = trash_left

        # Record metrics for each group
        for cfg_idx, (gamma, clip, envs, agents) in enumerate(config_groups):
            base = cfg_idx * agents_per_config
            key = (gamma, clip)
            for run in range(agents_per_config):
                i = base + run
                metrics[key]['episode_rewards'][run].append(ep_rewards[i])
                metrics[key]['episode_lengths'][run].append(ep_lengths[i])
                metrics[key]['episode_illegals'][run].append(ep_illegals[i])
                freq = ep_picked[i] / ep_lengths[i] if ep_lengths[i] > 0 else 0
                metrics[key]['episode_trash_picking_frequencies'][run].append(freq)
                metrics[key]['episode_trash_pieces_left'][run].append(ep_left[i])
                running_rewards[i] = 0.95 * running_rewards[i] + 0.05 * ep_rewards[i]

        # Save periodically
        if (episode + 1) % save_interval == 0:
            print(f"\nEpisode {episode + 1}")
            for cfg_idx, (gamma, clip, _, agents) in enumerate(config_groups):
                base = cfg_idx * agents_per_config
                avg_run = np.mean(running_rewards[base:base + agents_per_config])
                print(f"Config gamma={gamma}, clip={clip} -> Avg running reward: {avg_run:.2f}")
                for run, agent in enumerate(agents):
                    path = os.path.join(save_dir, f"ppo_gamma{gamma}_clip{clip}_run{run}.pt")
                    torch.save({
                        'episode': episode,
                        'model_state_dict': agent.actor_critic.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'running_reward': running_rewards[base + run],
                        'gamma': gamma,
                        'clip': clip
                    }, path)
            if episode >= 100:
                plot_moving_averages(metrics, window_size=100)

    return metrics

# plotting functions remain unchanged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000000)
    parser.add_argument("--agents-per-config", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--update-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--window-size", type=int, default=80)
    args = parser.parse_args()

    env_config = EnvConfig(max_steps=800)
    gamma_values = [0.99, 0.95, 0.9]
    clip_values = [0.2, 0.1, 0.3]
    gamma_and_clip_values = list(product(gamma_values, clip_values))

    metrics = train_multiple_ppo(
        gamma_and_clip_values=gamma_and_clip_values,
        env_config=env_config,
        agents_per_config=args.agents_per_config,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        render=args.render,
        save_interval=args.save_interval,
        save_dir=args.save_dir
    )

    plot_training_curves(metrics)
    plot_moving_averages(metrics, window_size=args.window_size)
