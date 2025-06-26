"""
final_experiment.py

Comprehensive experiment runner for DQN and PPO agents across all environment settings.
Generates detailed statistics, comparison plots, and saves results.

Author: Assistant & User
Date: 2025
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our evaluation functions
from simple_evaluate_ppo import simple_evaluate_ppo, PPOCompatibilityWrapper
from env.festival_trash_env import FestivalEnv, EnvConfig
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from tqdm import tqdm

def evaluate_agent(agent, env_config, episodes=50, max_steps=100, agent_type="dqn"):
    """Evaluate an agent (DQN or PPO) on a given environment configuration."""
    try:
        print("Creating environment...")
        env = FestivalEnv(cfg=env_config)
        rewards = []
        illegal_moves = []

        print(f"Evaluating {agent_type.upper()} for {episodes} episodes (max {max_steps} steps each)...")
        for episode in tqdm(range(episodes)):
            obs, _ = env.reset()
            done = truncated = False
            episode_reward = 0
            episode_illegal_moves = 0
            step_count = 0
            
            while not (done or truncated) and step_count < max_steps:
                try:
                    if agent_type.lower() == "ppo":
                        # PPO wrapper returns (action, log_prob, value)
                        if hasattr(agent, 'agent'):  # This is our PPOCompatibilityWrapper
                            action, _, _ = agent.act(obs)
                        else:
                            # Try to call act and handle different return formats
                            result = agent.act(obs)
                            if isinstance(result, tuple) and len(result) >= 3:
                                action, _, _ = result
                            else:
                                action = result
                    else:
                        # DQN agent returns just action
                        action = agent.act(obs)
                except Exception as e:
                    print(f"\nError in episode {episode}, step {step_count}: {e}")
                    break
                
                obs, reward, done, truncated, info, _ = env.step(action)
                episode_reward += reward
                if info.get("illegal", False):
                    episode_illegal_moves += 1
                step_count += 1
            
            rewards.append(episode_reward)
            illegal_moves.append(episode_illegal_moves)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_illegal = np.mean(illegal_moves)
        
        del env  # Cleanup
        return mean_reward, std_reward, mean_illegal
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None, None, None

# Backward compatibility
def evaluate_dqn(agent, env_config, episodes=50, max_steps=100):
    """Backward compatibility wrapper for DQN evaluation."""
    return evaluate_agent(agent, env_config, episodes, max_steps, "dqn")

class ExperimentRunner:
    """Main class for running comprehensive agent experiments."""
    
    def __init__(self, dqn_model_path: str, ppo_model_path: str, episodes: int = 50, max_steps: int = 100):
        self.dqn_model_path = dqn_model_path
        self.ppo_model_path = ppo_model_path
        self.episodes = episodes
        self.max_steps = max_steps
        self.results = {}
        
        # Environment configurations
        self.env_configs = {
            'Normal': EnvConfig(static_trash=True, static_benches=True),
            'Stochastic Benches': EnvConfig(static_trash=True, static_benches=False),
            'Stochastic Trash': EnvConfig(static_trash=False, static_benches=True),
            'Fully Stochastic': EnvConfig(static_trash=False, static_benches=False)
        }
        
        print(f"🎯 Experiment Configuration:")
        print(f"   Episodes per test: {episodes}")
        print(f"   Max steps per episode: {max_steps}")
        print(f"   Environment variants: {len(self.env_configs)}")
        print(f"   Total evaluations: {len(self.env_configs) * 2} (DQN + PPO)")
        
    def load_dqn_agent(self) -> DQNAgent:
        """Load and configure DQN agent."""
        print(f"📦 Loading DQN agent from: {self.dqn_model_path}")
        agent = DQNAgent(n_actions=8)
        agent.load(Path(self.dqn_model_path))
        agent.epsilon = 0.0  # No exploration during evaluation
        print("✅ DQN agent loaded successfully")
        return agent
        
    def load_ppo_agent(self) -> PPOCompatibilityWrapper:
        """Load and configure PPO agent with compatibility wrapper."""
        print(f"📦 Loading PPO agent from: {self.ppo_model_path}")
        
        # Create action space and agent with correct architecture
        action_space = type('ActionSpace', (), {'n': 9})()
        agent = PPOAgent(
            action_space=action_space,
            img_channels=1,  # Match saved model
            img_size=64,
            state_dim=4,
            hidden_dim=12,  # Match saved model
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            train_iters=10,
            batch_size=64,
        )
        
        agent.load(self.ppo_model_path)
        wrapper = PPOCompatibilityWrapper(agent)
        print("✅ PPO agent loaded successfully (with compatibility wrapper)")
        return wrapper
        
    def run_dqn_experiments(self) -> Dict:
        """Run DQN experiments across all environments."""
        print(f"\n{'='*60}")
        print("🤖 RUNNING DQN EXPERIMENTS")
        print(f"{'='*60}")
        
        dqn_agent = self.load_dqn_agent()
        dqn_results = {}
        
        for env_name, env_config in self.env_configs.items():
            print(f"\n🌍 Testing DQN on {env_name} environment...")
            start_time = time.time()
            
            mean_reward, std_reward, mean_illegal = evaluate_agent(
                dqn_agent, env_config, self.episodes, self.max_steps, "dqn"
            )
            
            elapsed = time.time() - start_time
            dqn_results[env_name] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_illegal': mean_illegal,
                'time_taken': elapsed
            }
            
            print(f"   ✅ Results: {mean_reward:.2f} ± {std_reward:.2f} (illegal: {mean_illegal:.1f}) [{elapsed:.1f}s]")
            
        print(f"\n✅ DQN experiments completed!")
        return dqn_results
        
    def run_ppo_experiments(self) -> Dict:
        """Run PPO experiments across all environments."""
        print(f"\n{'='*60}")
        print("🎭 RUNNING PPO EXPERIMENTS")
        print(f"{'='*60}")
        
        ppo_agent = self.load_ppo_agent()
        ppo_results = {}
        
        for env_name, env_config in self.env_configs.items():
            print(f"\n🌍 Testing PPO on {env_name} environment...")
            start_time = time.time()
            
            mean_reward, std_reward, mean_illegal = evaluate_agent(
                ppo_agent, env_config, self.episodes, self.max_steps, "ppo"
            )
            
            elapsed = time.time() - start_time
            ppo_results[env_name] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_illegal': mean_illegal,  # Now tracked for PPO too
                'time_taken': elapsed
            }
            
            print(f"   ✅ Results: {mean_reward:.2f} ± {std_reward:.2f} [{elapsed:.1f}s]")
            
        print(f"\n✅ PPO experiments completed!")
        return ppo_results
        
    def run_all_experiments(self) -> Dict:
        """Run all experiments and collect results."""
        print(f"\n🚀 STARTING COMPREHENSIVE AGENT EVALUATION")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Run experiments
        dqn_results = self.run_dqn_experiments()
        ppo_results = self.run_ppo_experiments()
        
        total_time = time.time() - start_time
        
        # Combine results
        self.results = {
            'DQN': dqn_results,
            'PPO': ppo_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'episodes_per_test': self.episodes,
                'max_steps_per_episode': self.max_steps,
                'total_time': total_time,
                'dqn_model': self.dqn_model_path,
                'ppo_model': self.ppo_model_path
            }
        }
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Total episodes run: {self.episodes * len(self.env_configs) * 2}")
        
        return self.results
        
    def create_comparison_plots(self, save_dir: str = "experiment_results"):
        """Create comprehensive comparison plots."""
        print(f"\n📊 Creating comparison plots...")
        
        Path(save_dir).mkdir(exist_ok=True)
        
        # Prepare data for plotting
        agents = []
        environments = []
        mean_rewards = []
        std_rewards = []
        
        for agent_type in ['DQN', 'PPO']:
            for env_name in self.env_configs.keys():
                agents.append(agent_type)
                environments.append(env_name)
                mean_rewards.append(self.results[agent_type][env_name]['mean_reward'])
                std_rewards.append(self.results[agent_type][env_name]['std_reward'])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bar plot comparison
        x_pos = np.arange(len(agents))
        colors = ['#1f77b4' if agent == 'DQN' else '#ff7f0e' for agent in agents]
        
        bars = ax1.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, color=colors, alpha=0.7)
        ax1.set_xlabel('Agent-Environment Combination')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Agent Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{a}\n{e}" for a, e in zip(agents, environments)], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, mean_rewards, std_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Heatmap
        env_names = list(self.env_configs.keys())
        agent_names = ['DQN', 'PPO']
        
        heatmap_data = np.array([
            [self.results[agent][env]['mean_reward'] for env in env_names]
            for agent in agent_names
        ])
        
        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(range(len(env_names)))
        ax2.set_yticks(range(len(agent_names)))
        ax2.set_xticklabels(env_names, rotation=45, ha='right')
        ax2.set_yticklabels(agent_names)
        ax2.set_title('Performance Heatmap')
        
        # Add text annotations
        for i in range(len(agent_names)):
            for j in range(len(env_names)):
                ax2.text(j, i, f'{heatmap_data[i, j]:.1f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Mean Reward')
        
        # 3. Environment difficulty analysis
        env_difficulty = {}
        for env_name in env_names:
            dqn_reward = self.results['DQN'][env_name]['mean_reward']
            ppo_reward = self.results['PPO'][env_name]['mean_reward']
            avg_reward = (dqn_reward + ppo_reward) / 2
            env_difficulty[env_name] = avg_reward
        
        sorted_envs = sorted(env_difficulty.items(), key=lambda x: x[1], reverse=True)
        env_names_sorted = [x[0] for x in sorted_envs]
        difficulty_scores = [x[1] for x in sorted_envs]
        
        ax3.barh(env_names_sorted, difficulty_scores, color='skyblue', alpha=0.7)
        ax3.set_xlabel('Average Performance (Both Agents)')
        ax3.set_title('Environment Difficulty Ranking')
        ax3.grid(True, alpha=0.3)
        
        # 4. Agent comparison radar chart
        categories = [env.replace(' ', '\n') for env in env_names]
        
        # Normalize rewards to 0-1 scale for radar chart
        all_rewards = []
        for agent in ['DQN', 'PPO']:
            for env in env_names:
                all_rewards.append(self.results[agent][env]['mean_reward'])
        
        min_reward, max_reward = min(all_rewards), max(all_rewards)
        
        def normalize_reward(reward):
            if max_reward == min_reward:
                return 0.5
            return (reward - min_reward) / (max_reward - min_reward)
        
        dqn_normalized = [normalize_reward(self.results['DQN'][env]['mean_reward']) for env in env_names]
        ppo_normalized = [normalize_reward(self.results['PPO'][env]['mean_reward']) for env in env_names]
        
        # Simple line plot instead of radar (easier to read)
        ax4.plot(categories, dqn_normalized, 'o-', label='DQN', linewidth=2, markersize=8)
        ax4.plot(categories, ppo_normalized, 's-', label='PPO', linewidth=2, markersize=8)
        ax4.set_ylabel('Normalized Performance')
        ax4.set_title('Agent Performance Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(save_dir) / f"agent_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Plots saved to: {plot_path}")
        
        plt.show()
        
    def save_results(self, save_dir: str = "experiment_results"):
        """Save detailed results to files."""
        print(f"\n💾 Saving results...")
        
        Path(save_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = Path(save_dir) / f"experiment_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"   📄 JSON results: {json_path}")
        
        # Save CSV summary
        csv_data = []
        for agent_type in ['DQN', 'PPO']:
            for env_name, result in self.results[agent_type].items():
                csv_data.append({
                    'Agent': agent_type,
                    'Environment': env_name,
                    'Mean_Reward': result['mean_reward'],
                    'Std_Reward': result['std_reward'],
                    'Mean_Illegal': result['mean_illegal'],
                    'Time_Taken': result['time_taken']
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = Path(save_dir) / f"experiment_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   📊 CSV summary: {csv_path}")
        
        # Save formatted report
        report_path = Path(save_dir) / f"experiment_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE AGENT EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {self.results['metadata']['timestamp']}\n")
            f.write(f"Episodes per test: {self.results['metadata']['episodes_per_test']}\n")
            f.write(f"Max steps per episode: {self.results['metadata']['max_steps_per_episode']}\n")
            f.write(f"Total evaluation time: {self.results['metadata']['total_time']:.1f} seconds\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-"*30 + "\n\n")
            
            for agent_type in ['DQN', 'PPO']:
                f.write(f"{agent_type} AGENT RESULTS:\n")
                for env_name, result in self.results[agent_type].items():
                    f.write(f"  {env_name:20}: {result['mean_reward']:6.2f} ± {result['std_reward']:5.2f} ")
                    f.write(f"(illegal: {result['mean_illegal']:4.1f}) [{result['time_taken']:5.1f}s]\n")
                f.write("\n")
                
        print(f"   📋 Report: {report_path}")
        
    def print_summary(self):
        """Print a comprehensive summary of results."""
        print(f"\n{'='*80}")
        print("📊 EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        
        for agent_type in ['DQN', 'PPO']:
            print(f"\n🤖 {agent_type} AGENT PERFORMANCE:")
            print(f"   {'Environment':<20} {'Mean Reward':<12} {'Std Dev':<10} {'Illegal Moves':<12}")
            print(f"   {'-'*60}")
            
            for env_name, result in self.results[agent_type].items():
                print(f"   {env_name:<20} {result['mean_reward']:8.2f}     "
                      f"{result['std_reward']:6.2f}     {result['mean_illegal']:8.1f}")
        
        print(f"\n🏆 BEST PERFORMANCES:")
        
        # Find best performance for each environment
        for env_name in self.env_configs.keys():
            dqn_reward = self.results['DQN'][env_name]['mean_reward']
            ppo_reward = self.results['PPO'][env_name]['mean_reward']
            
            if dqn_reward > ppo_reward:
                winner = "DQN"
                margin = dqn_reward - ppo_reward
            else:
                winner = "PPO"
                margin = ppo_reward - dqn_reward
                
            print(f"   {env_name:<20}: {winner} (+{margin:5.2f})")

def main():
    """Main experiment execution function."""
    parser = argparse.ArgumentParser(description='Comprehensive agent evaluation experiments')
    parser.add_argument('--dqn_model', type=str, 
                      default='final_models/dqn_model_2025-06-25__14-20-53.pt',
                      help='Path to DQN model')
    parser.add_argument('--ppo_model', type=str, 
                      default='final_models/ppo_agent1.pt',
                      help='Path to PPO model')
    parser.add_argument('--episodes', type=int, default=50,
                      help='Number of episodes per evaluation')
    parser.add_argument('--max_steps', type=int, default=100,
                      help='Maximum steps per episode')
    parser.add_argument('--save_dir', type=str, default='experiment_results',
                      help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                      help='Quick test with fewer episodes')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick:
        episodes = 10
        max_steps = 50
        print("🚀 Running QUICK evaluation mode")
    else:
        episodes = args.episodes
        max_steps = args.max_steps
        print("🚀 Running FULL evaluation mode")
    
    # Initialize and run experiments
    runner = ExperimentRunner(
        dqn_model_path=args.dqn_model,
        ppo_model_path=args.ppo_model,
        episodes=episodes,
        max_steps=max_steps
    )
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Generate outputs
    runner.print_summary()
    runner.save_results(args.save_dir)
    runner.create_comparison_plots(args.save_dir)
    
    print(f"\n🎉 EXPERIMENT COMPLETE!")
    print(f"   Check '{args.save_dir}' directory for detailed results and plots")

if __name__ == "__main__":
    main() 