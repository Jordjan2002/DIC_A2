"""
Run episodes and visualise the festival field with trees, trash, bins and the robot path.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from tqdm import tqdm
from pathlib import Path

from env import FestivalEnv, EnvConfig
from env.simple_festival_env import SimpleFestivalEnv
from agents import RandomAgent
from agents.dqn_agent import DQNAgent
from typing import TYPE_CHECKING
import time
if TYPE_CHECKING:                 # only for type checkers / IDEs
    from env.stochastic_festival_trash_env import FestivalEnvPeople


def get_latest_model():
    """Get the path to the most recently trained model."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    model_files = list(results_dir.glob("dqn_model_*.pt"))
    if not model_files:
        return None
    return str(max(model_files, key=lambda p: p.stat().st_mtime))


class FieldRenderer:
    def __init__(self, cfg: EnvConfig, show_people: bool = False):
        self.cfg = cfg
        self.show_people = show_people          # NEW
        self.fig, self.ax = plt.subplots(figsize=(6, 10))
        self._setup_base()


    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _setup_base(self):
        """Initialize the start image on a cleared axis."""
        self.ax.clear()
        self.ax.set_xlim(0, self.cfg.width)
        self.ax.set_ylim(self.cfg.height, 0)
        self.ax.set_aspect("equal")

        # field borders
        self.ax.add_patch(
            patches.Rectangle((0, 0), self.cfg.width, self.cfg.height, fill=False, lw=2)
        )
        # notch (light grey square)
        self.ax.add_patch(
            patches.Rectangle(
                (self.cfg.notch_x0, self.cfg.notch_y0),
                self.cfg.notch_x1 - self.cfg.notch_x0,
                self.cfg.notch_y1 - self.cfg.notch_y0,
                color="lightgrey",
                alpha=0.3,
            )
        )

        # robot (blue dot)
        self.robot_patch = patches.Circle((0, 0), self.cfg.robot_radius, color="blue")
        self.ax.add_patch(self.robot_patch)

        # robot sensor area (transparent yellow box)
        crop = self.cfg.crop_size
        self.sensor_patch = patches.Rectangle(
            (0, 0), crop, crop,
            color="yellow",
            alpha=0.2,
            zorder=1
        )
        self.ax.add_patch(self.sensor_patch)

        # add textbox for current load
        self.load_text = self.ax.text(
            0.5,           
            1.02,          
            "",            
            transform=self.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

        # initialize trash, tree and bin patches
        self.trash_patches, self.tree_patches, self.bin_patches = [], [], []

        # initialize people patches
        self.person_patches = [] if self.show_people else None



    # call at every episode
    def reset_axes(self):
        self._setup_base()

    def init_static(self, env: FestivalEnv):
        """ Add objects to the drawn environment that are there when the robot starts """
        # trees
        for tx, ty in env.trees:
            circ = patches.Circle((tx, ty), radius=env.cfg.tree_radius, color="green")
            self.ax.add_patch(circ)
            self.tree_patches.append(circ)
        # bins
        bin_w = 1.5
        bin_h = 1.5
        for bx, by in env.bins:
            rect = patches.Rectangle(
                (bx - bin_w / 2, by - bin_h / 2),  # linkerbovenhoek
                bin_w,
                bin_h,
                color="black"
            )
            self.ax.add_patch(rect)
            self.bin_patches.append(rect)

        # yellow benches
        for cx, cy, w, h, deg in env.rect_obs:
            rect = patches.Rectangle((cx - w/2, cy - h/2), w, h,angle=deg, color="yellow", alpha=0.7, rotation_point="center")
            self.ax.add_patch(rect)
        
        # add people that are already on the field when the robot begins
        if self.show_people and hasattr(env, "people"): #if self.show_people:
            for p in env.people:                         # may be empty at t=0
                circ = patches.Circle((p.x, p.y), 0.4, color="purple")
                self.ax.add_patch(circ)
                self.person_patches.append(circ)


    def init_trash(self, env: FestivalEnv):
        """ Add the trash to the drawn environment """
        for tx, ty in env.trash:
            circ = patches.Circle(
                (tx, ty), radius=env.cfg.trash_radius, color="red", alpha=0.8
            )
            self.ax.add_patch(circ)
            self.trash_patches.append(circ)

    def update(self, env: FestivalEnv, illegal: bool = False):
        """ Update the visualisation to reflect the current state of the environment """

        # If people is enabled, add and remove people
        if self.show_people and hasattr(env, "people"):
            # add missing patches
            while len(self.person_patches) < len(env.people):
                circ = patches.Circle((0, 0), 0.4, color="purple")
                self.ax.add_patch(circ)
                self.person_patches.append(circ)

            # delete surplus patches
            while len(self.person_patches) > len(env.people):
                circ = self.person_patches.pop()
                circ.remove()                      # fully remove from the axes

        # update robot and sensor position
        self.robot_patch.center = (env.x, env.y)
        half = self.cfg.crop_size / 2
        self.sensor_patch.set_xy((env.x - half, env.y - half))

        # update picked up trash
        for circ, alive in zip(self.trash_patches, env.trash_mask):
            circ.set_visible(alive)

        # Show text when the robot collides
        if illegal:
            self.load_text.set_text("Collision penalty!" if illegal else "")
        else:
            self.load_text.set_text(f"Load: {env.load} / {env.cfg.max_load}")

        # Update the people positions
        if self.show_people and hasattr(env, "people"): #if self.show_people:
            # add patches if new people appeared
            while len(self.person_patches) < len(env.people):
                circ = patches.Circle((0, 0), 0.4, color="purple")
                self.ax.add_patch(circ)
                self.person_patches.append(circ)

            # update positions
            for circ, person in zip(self.person_patches, env.people):
                circ.center = (person.x, person.y)

            # hide surplus circles (people that left)
            for circ in self.person_patches[len(env.people):]:
                circ.set_visible(False)


        plt.pause(0.001)
        

def run_episode(env: FestivalEnv, agent, renderer: FieldRenderer | None = None):
    """ Run a single episode of the environment """
    obs, _ = env.reset()

    # Initialize renderer
    if renderer is not None:
        renderer.reset_axes()
        renderer.init_static(env)
        renderer.init_trash(env)
        renderer.update(env)

    # Initialize counters    
    total_reward = 0.0
    illegal_moves = 0

    # Main episode loop
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, trunc, info = env.step(action)
        done = done or trunc  
        total_reward += reward
        if info.get("illegal", False):
            illegal_moves += 1
        if renderer is not None:
            renderer.update(env, illegal=info.get("illegal", False))
    return total_reward, illegal_moves


if __name__ == "__main__":
    # Define possible arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to the trained model file. If not specified, uses the most recent model.")
    args = parser.parse_args()

    # Get the model path
    model_path = args.model if args.model else get_latest_model()
    if not model_path:
        print("No trained model found. Please train a model first.")
        exit(1)
    print(f"Using model: {model_path}")

    cfg = EnvConfig()
    env = FestivalEnv(cfg)
    
    # Initialize and load the trained DQN agent
    agent = DQNAgent()
    agent.load(model_path)

    renderer = (FieldRenderer(cfg, show_people=args.people_env) if args.render else None)

    # Run episodes
    rewards, illegals = [], []
    for _ in tqdm(range(args.episodes), desc="episodes"):
        r, ill = run_episode(env, agent, renderer)
        rewards.append(r)
        illegals.append(ill)

    env.close()
    if args.render:
        plt.show(block=True)

    # Print results
    print("mean reward:", np.mean(rewards))
    print("mean illegal moves per ep:", np.mean(illegals))