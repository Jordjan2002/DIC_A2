import argparse
import matplotlib.pyplot as plt
from agents.ppo_agent import PPOAgent
from env.simple_festival_env import SimpleFestivalEnv, SimpleEnvConfig
from env.festival_trash_env import FestivalEnv, EnvConfig
from simulation import run_episode, FieldRenderer

if __name__ == "__main__":
    env = FestivalEnv(seed=42, cfg=EnvConfig(max_steps=10000))
    renderer = FieldRenderer(env.cfg, show_people=False)
    # Load the trained model
    agent = PPOAgent(
        action_space=env.action_space,
        img_channels=1,  # 1 channel for the sensor image
        img_size=64,     # 64x64 image size
        state_dim=4,     # 4-dimensional state vector
        hidden_dim=12,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        train_iters=10,
        batch_size=32
    )
    agent.load("models/ppo_gamma0.99_clip0.2_run1.pt")
    renderer.reset_axes()
    renderer.init_static(env)
    renderer.init_trash(env)
    total_reward, illegal_moves = run_episode(env, agent, renderer)
    print(f"Total Reward: {total_reward}, Illegal Moves: {illegal_moves}")
    plt.show()  # Keep the plot open until closed by the user
    renderer.close()  # Close the renderer properly