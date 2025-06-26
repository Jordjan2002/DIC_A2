#!/usr/bin/env python3
"""
run_comprehensive_experiments.py

Comprehensive experiment runner for DQN and PPO agents across different environments.
Bypasses memory issues by using direct environment interaction.

Usage:
    python run_comprehensive_experiments.py --runs 5
    python run_comprehensive_experiments.py --runs 3 --no-plots
"""

import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simple_direct_test(env_config, agent_type="dqn", runs=5, verbose=True):
    """Direct environment interaction without the complex evaluation function."""
    from env.festival_trash_env import FestivalEnv
    from agents.dqn_agent import DQNAgent
    from agents.ppo_agent import PPOAgent
    from simple_evaluate_ppo import PPOCompatibilityWrapper
    
    rewards = []
    
    for run in range(runs):
        try:
            # Load agent
            if agent_type == "dqn":
                agent = DQNAgent(n_actions=8)
                agent.load(Path("final_models/dqn_model_2025-06-25__14-20-53.pt"))
                agent.epsilon = 0.0
            else:
                action_space = type('ActionSpace', (), {'n': 9})()
                ppo_agent = PPOAgent(
                    action_space=action_space, img_channels=1, img_size=64, state_dim=4,
                    hidden_dim=12, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                    train_iters=10, batch_size=64
                )
                ppo_agent.load("final_models/ppo_agent1.pt")
                agent = PPOCompatibilityWrapper(ppo_agent)
            
            # Create environment
            env = FestivalEnv(cfg=env_config)
            obs, _ = env.reset()
            
            total_reward = 0
            for step in range(10):
                # Get action
                if agent_type == "dqn":
                    action = agent.act(obs)
                else:
                    action, _, _ = agent.act(obs)
                
                # Take step
                obs, reward, done, truncated, info, remaining = env.step(action)
                total_reward += reward
                
                if done or truncated:
                    break
            
            rewards.append(total_reward)
            if verbose:
                print(f"  Run {run+1}: {total_reward:.2f}")
            
            # Cleanup
            env.close()
            del env, agent
            if agent_type == "ppo":
                del ppo_agent
            gc.collect()
            
        except Exception as e:
            print(f"  Run {run+1} failed: {e}")
            rewards.append(None)
    
    valid_rewards = [r for r in rewards if r is not None]
    if valid_rewards:
        mean_reward = sum(valid_rewards) / len(valid_rewards)
        return mean_reward, valid_rewards
    else:
        return None, []

def test_environment(env_name, env_config, runs=5):
    """Test both agents on a single environment."""
    print(f"\n🧪 {env_name.upper()} ENVIRONMENT ({runs} runs each)")
    print("="*60)
    
    # Test DQN
    print("🤖 DQN:")
    dqn_mean, dqn_rewards = simple_direct_test(env_config, "dqn", runs)
    
    # Test PPO
    print("🎭 PPO:")
    ppo_mean, ppo_rewards = simple_direct_test(env_config, "ppo", runs)
    
    # Results
    print(f"\n📊 {env_name.upper()} RESULTS:")
    print(f"DQN: {dqn_mean:.2f} (runs: {dqn_rewards})")
    print(f"PPO: {ppo_mean:.2f} (runs: {ppo_rewards})")
    winner = "DQN" if dqn_mean > ppo_mean else "PPO"
    print(f"Winner: {winner}")
    
    return {
        'DQN': {'mean': dqn_mean, 'rewards': dqn_rewards},
        'PPO': {'mean': ppo_mean, 'rewards': ppo_rewards}
    }

def run_all_experiments(runs=5):
    """Run experiments on all environments."""
    from env.festival_trash_env import EnvConfig
    
    print(f"🎯 COMPREHENSIVE EXPERIMENT SUITE ({runs} runs per agent)")
    print("="*70)
    
    # Environment configurations
    env_configs = {
        'Normal': EnvConfig(static_trash=True, static_benches=True),
        'Stochastic_Benches': EnvConfig(static_trash=True, static_benches=False),
        'Stochastic_Trash': EnvConfig(static_trash=False, static_benches=True),
    }
    
    results_data = {}
    
    # Run tests for each environment
    for env_name, env_config in env_configs.items():
        results_data[env_name] = test_environment(env_name, env_config, runs)
        # Force cleanup between environments
        gc.collect()
        time.sleep(1)  # Small delay to help with cleanup
    
    return results_data

def analyze_results(results_data, runs):
    """Analyze and print statistical results."""
    print(f"\n🏆 COMPREHENSIVE EXPERIMENT RESULTS ({runs} runs per agent)")
    print("="*85)
    print(f"{'Environment':<20} {'DQN Mean':<12} {'DQN Std':<10} {'PPO Mean':<12} {'PPO Std':<10} {'Winner':<8} {'p-value'}")
    print("-" * 85)
    
    dqn_wins = ppo_wins = 0
    
    for env_name, data in results_data.items():
        dqn_rewards = data['DQN']['rewards']
        ppo_rewards = data['PPO']['rewards']
        dqn_mean = data['DQN']['mean']
        dqn_std = np.std(dqn_rewards) if len(dqn_rewards) > 1 else 0
        ppo_mean = data['PPO']['mean'] 
        ppo_std = np.std(ppo_rewards) if len(ppo_rewards) > 1 else 0
        
        # Statistical significance test (t-test)
        if len(dqn_rewards) > 1 and len(ppo_rewards) > 1:
            t_stat, p_value = stats.ttest_ind(dqn_rewards, ppo_rewards)
        else:
            p_value = 1.0
        
        winner = "DQN" if dqn_mean > ppo_mean else "PPO"
        significance = "*" if p_value < 0.05 else ""
        
        if winner == "DQN":
            dqn_wins += 1
        else:
            ppo_wins += 1
        
        print(f"{env_name:<20} {dqn_mean:<12.2f} {dqn_std:<10.2f} {ppo_mean:<12.2f} {ppo_std:<10.2f} {winner:<8} {p_value:.3f}{significance}")
    
    print(f"\n🎯 FINAL SCORE: DQN {dqn_wins} - {ppo_wins} PPO")
    print("* indicates statistically significant difference (p < 0.05)")
    
    # Print detailed statistics
    print(f"\n📈 DETAILED STATISTICS:")
    for env_name, data in results_data.items():
        print(f"\n{env_name}:")
        dqn_r = data['DQN']['rewards']
        ppo_r = data['PPO']['rewards']
        print(f"  DQN: mean={np.mean(dqn_r):.2f}, std={np.std(dqn_r):.2f}, min={min(dqn_r):.2f}, max={max(dqn_r):.2f}")
        print(f"  PPO: mean={np.mean(ppo_r):.2f}, std={np.std(ppo_r):.2f}, min={min(ppo_r):.2f}, max={max(ppo_r):.2f}")

def create_visualizations(results_data, runs, save_plots=True):
    """Create comprehensive visualization plots."""
    try:
        import matplotlib.pyplot as plt
        
        # Create enhanced comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Agent Performance Comparison Across 3 Environments ({runs} runs each)', fontsize=16)
        
        environments = list(results_data.keys())
        dqn_means = [results_data[env]['DQN']['mean'] for env in environments]
        ppo_means = [results_data[env]['PPO']['mean'] for env in environments]
        dqn_stds = [np.std(results_data[env]['DQN']['rewards']) for env in environments]
        ppo_stds = [np.std(results_data[env]['PPO']['rewards']) for env in environments]
        
        # Plot 1: Bar chart with error bars
        x = np.arange(len(environments))
        width = 0.35
        bars1 = ax1.bar(x - width/2, dqn_means, width, label='DQN', yerr=dqn_stds, capsize=5, alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, ppo_means, width, label='PPO', yerr=ppo_stds, capsize=5, alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Mean Performance Comparison (± 1 std)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([env.replace('_', '\n') for env in environments], fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plots for each environment
        all_data = []
        labels = []
        colors = []
        for i, env in enumerate(environments):
            dqn_runs = results_data[env]['DQN']['rewards']
            ppo_runs = results_data[env]['PPO']['rewards']
            all_data.extend([dqn_runs, ppo_runs])
            labels.extend([f"{env.replace('_', ' ')}\nDQN", f"{env.replace('_', ' ')}\nPPO"])
            colors.extend(['skyblue', 'lightcoral'])
        
        bp = ax2.boxplot(all_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Reward')
        ax2.set_title('Distribution of Individual Runs')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance difference
        differences = [dqn_means[i] - ppo_means[i] for i in range(len(environments))]
        colors = ['green' if d > 0 else 'red' for d in differences]
        bars = ax3.bar(environments, differences, color=colors, alpha=0.7)
        ax3.set_xlabel('Environment')
        ax3.set_ylabel('DQN - PPO (Reward Difference)')
        ax3.set_title('Performance Gap Analysis')
        ax3.set_xticklabels([env.replace('_', '\n') for env in environments], fontsize=10)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        # Add value labels
        for bar, diff in zip(bars, differences):
            ax3.text(bar.get_x() + bar.get_width()/2., diff + (0.1 if diff > 0 else -0.1), 
                     f'{diff:.2f}', ha='center', va='bottom' if diff > 0 else 'top', fontsize=10)
        
        # Plot 4: Individual run scatter plot
        for i, env in enumerate(environments):
            dqn_runs = results_data[env]['DQN']['rewards']
            ppo_runs = results_data[env]['PPO']['rewards']
            ax4.scatter([i-0.1]*len(dqn_runs), dqn_runs, label='DQN' if i==0 else '', alpha=0.7, s=60, color='skyblue')
            ax4.scatter([i+0.1]*len(ppo_runs), ppo_runs, label='PPO' if i==0 else '', alpha=0.7, s=60, color='lightcoral')
            # Add mean lines
            ax4.hlines(np.mean(dqn_runs), i-0.2, i, colors='blue', alpha=0.8, linewidth=3)
            ax4.hlines(np.mean(ppo_runs), i, i+0.2, colors='red', alpha=0.8, linewidth=3)
        
        ax4.set_xlabel('Environment')
        ax4.set_ylabel('Reward')
        ax4.set_title('Individual Run Results with Means')
        ax4.set_xticks(range(len(environments)))
        ax4.set_xticklabels([env.replace('_', '\n') for env in environments], fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'agent_comparison_{runs}runs.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Plots saved to 'agent_comparison_{runs}runs.png'")
        
        plt.show()
        
    except ImportError:
        print("\n⚠️  Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"\n⚠️  Error creating plots: {e}")

def save_results(results_data, runs):
    """Save results to JSON file."""
    # Add metadata
    results_with_metadata = {
        'metadata': {
            'runs_per_agent': runs,
            'total_environments': len(results_data),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'episodes_per_run': 1,
            'steps_per_episode': 10
        },
        'results': results_data
    }
    
    filename = f'experiment_results_{runs}runs.json'
    with open(filename, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    print(f"\n💾 Results saved to '{filename}'")

def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run comprehensive agent experiments')
    parser.add_argument('--runs', type=int, default=5, 
                       help='Number of runs per agent per environment (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to file (default: True)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting comprehensive experiments with {args.runs} runs per agent")
    
    start_time = time.time()
    
    # Run all experiments
    results_data = run_all_experiments(args.runs)
    
    # Analyze results
    analyze_results(results_data, args.runs)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(results_data, args.runs, args.save_plots)
    
    # Save results
    save_results(results_data, args.runs)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total experiment time: {total_time:.1f} seconds")
    print("✅ Experiments completed successfully!")

if __name__ == "__main__":
    main() 