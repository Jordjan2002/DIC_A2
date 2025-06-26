"""
test_experiments.py

Quick test script to verify that the experiment file works with minimal parameters.
Runs for just 2 episodes with 10 steps each for both DQN and PPO agents.
"""
import sys
from pathlib import Path
from final_experiment import evaluate_agent
from simple_evaluate_ppo import simple_evaluate_ppo, PPOCompatibilityWrapper
from env.festival_trash_env import EnvConfig
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent

def test_dqn_experiments():
    """Test DQN experiments with minimal parameters."""
    print("=== Quick Test of DQN Experiment System ===")
    
    # Use the final DQN model
    model_path = "final_models/dqn_model_2025-06-25__14-20-53.pt"
    
    if not Path(model_path).exists():
        print(f"Error: DQN model file {model_path} not found!")
        return False
    
    print(f"Testing DQN with model: {model_path}")
    
    # Load DQN agent
    agent = DQNAgent(n_actions=8)
    agent.load(Path(model_path))
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Test each experiment setting
    env_configs = {
        'normal': EnvConfig(static_trash=True, static_benches=True),
        'benches': EnvConfig(static_trash=True, static_benches=False),
        'trash': EnvConfig(static_trash=False, static_benches=True)
    }
    
    for exp_name, env_config in env_configs.items():
        print(f"\n{'='*50}")
        print(f"Testing DQN {exp_name.upper()} setting")
        print(f"{'='*50}")
        
        try:
            mean_reward, std_reward, mean_illegal = evaluate_agent(agent, env_config, episodes=2, max_steps=10, agent_type="dqn")
            print(f"✅ DQN {exp_name}: {mean_reward:.2f} ± {std_reward:.2f} (illegal: {mean_illegal:.1f})")
        except Exception as e:
            print(f"❌ DQN {exp_name} experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_ppo_experiments():
    """Test PPO experiments with minimal parameters."""
    print("\n=== Quick Test of PPO Experiment System ===")
    
    # Use the final PPO model
    model_path = "final_models/ppo_agent1.pt"
    
    if not Path(model_path).exists():
        print(f"Error: PPO model file {model_path} not found!")
        return False
    
    print(f"Testing PPO with model: {model_path}")
    
    # Load PPO agent with compatibility wrapper
    action_space = type('ActionSpace', (), {'n': 9})()
    ppo_agent = PPOAgent(
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
    ppo_agent.load(model_path)
    agent_wrapper = PPOCompatibilityWrapper(ppo_agent)
    
    # Test each experiment setting
    env_configs = {
        'normal': EnvConfig(static_trash=True, static_benches=True),
        'benches': EnvConfig(static_trash=True, static_benches=False),
        'trash': EnvConfig(static_trash=False, static_benches=True)
    }
    
    for exp_name, env_config in env_configs.items():
        print(f"\n{'='*50}")
        print(f"Testing PPO {exp_name.upper()} setting")
        print(f"{'='*50}")
        
        try:
            mean_reward, std_reward, mean_illegal = evaluate_agent(agent_wrapper, env_config, episodes=2, max_steps=10, agent_type="ppo")
            print(f"✅ PPO {exp_name}: {mean_reward:.2f} ± {std_reward:.2f} (illegal: {mean_illegal:.1f})")
        except Exception as e:
            print(f"❌ PPO {exp_name} experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_experiments():
    """Test both DQN and PPO experiments with minimal parameters."""
    print("=== Quick Test of Complete Experiment System ===")
    
    # Test DQN experiments
    dqn_success = test_dqn_experiments()
    
    # Test PPO experiments
    ppo_success = test_ppo_experiments()
    
    print(f"\n{'='*50}")
    print("All tests completed!")
    print(f"{'='*50}")
    
    if dqn_success and ppo_success:
        print("✅ All experiments passed successfully!")
        print("🎯 Both DQN and PPO agents are working perfectly!")
    else:
        print("❌ Some experiments failed. Check the output above.")
        if dqn_success:
            print("✅ DQN experiments passed")
        if ppo_success:
            print("✅ PPO experiments passed")
    
    return dqn_success and ppo_success

if __name__ == "__main__":
    test_experiments() 