"""
Train a PPO agent on the festival environment.
"""

import argparse
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from env import EnvConfig, FestivalEnv
from agents import PPOAgent
from simulation import run_episode, FieldRenderer

def train_ppo(
    env,
    agent: PPOAgent,
    num_episodes: int,
    update_interval: int = 2048,  # Update policy every N steps
    render: bool = False,
    save_interval: int = 100,  # Save model every N episodes
    save_path: str = "ppo_model.pt"
):
    """Train the PPO agent."""
    renderer = FieldRenderer(env.cfg) if render else None
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_illegals = []
    running_reward = 0
    
    # Training loop
    total_steps = 0
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs, _ = env.reset()
        
        if render and renderer:
            renderer.reset_axes()
            renderer.init_static(env)
            renderer.init_trash(env)

        episode_reward = 0
        episode_length = 0
        episode_illegal = 0
        done = False
        
        # Episode loop
        while not done:
            # Get action and value
            with torch.no_grad():
                obs_tensor = {
                    'image': torch.FloatTensor(obs['image']).unsqueeze(0).to(agent.device),
                    'state': torch.FloatTensor(obs['state']).unsqueeze(0).to(agent.device)
                }
                action_probs, value = agent.actor_critic(obs_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Take action
            next_obs, reward, done, trunc, info = env.step(action.item())
            done = done or trunc
            
            # Store transition
            agent.store_transition(
                obs=obs,
                action=action.item(),
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done
            )
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            if info.get("illegal", False):
                episode_illegal += 1
            
            # Update observation
            obs = next_obs
            total_steps += 1

            # Render environment if enabled
            if render and renderer:
                renderer.update(env, illegal=info.get("illegal", False))
            
            # Update policy if enough steps collected
            if len(agent.obs_buffer) >= update_interval:
                agent.update()
        
        # Update metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_illegals.append(episode_illegal)
        running_reward = 0.95 * running_reward + 0.05 * episode_reward
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'running_reward': running_reward,
            }, save_path)
            print(f"\nEpisode {episode + 1}")
            print(f"Running reward: {running_reward:.2f}")
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"Episode length: {episode_length}")
            print(f"Illegal moves: {episode_illegal}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_illegals': episode_illegals,
        'running_reward': running_reward
    }

def plot_training_curves(metrics):
    """Plot training metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(metrics['episode_rewards'], label='Episode Reward')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    
    # Plot episode lengths
    ax2.plot(metrics['episode_lengths'], label='Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    
    # Plot illegal moves
    ax3.plot(metrics['episode_illegals'], label='Illegal Moves')
    ax3.set_title('Illegal Moves per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--update-interval", type=int, default=2048)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="ppo_model.pt")
    parser.add_argument("--load-path", type=str, default=None)
    args = parser.parse_args()
    
    # Create environment
    cfg = EnvConfig(max_steps=500)
    # env = FestivalEnv(cfg)
    env = FestivalEnv(render_mode="human")
    
    # Create agent
    agent = PPOAgent(
        action_space=env.action_space,
        img_channels=1,  # 1 channel for the sensor image
        img_size=64,     # 64x64 image size
        state_dim=4,     # 4-dimensional state vector
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        train_iters=10,
        batch_size=64
    )
    
    # Load pretrained model if specified
    if args.load_path:
        checkpoint = torch.load(args.load_path)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model from {args.load_path}")
    
    # Train agent
    metrics = train_ppo(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        render=args.render,
        save_interval=args.save_interval,
        save_path=args.save_path
    )
    
    # Plot training curves
    plot_training_curves(metrics)
    
    env.close()