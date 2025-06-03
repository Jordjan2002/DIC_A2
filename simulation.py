"""
Run episodes and visualise the festival field with trees, trash, bins and the robot path.
Usage:
    python simulate.py --episodes 5 --render
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from tqdm import tqdm

from env import FestivalEnv, EnvConfig
# from agents import RandomAgent
from agents.newppo import PPOAgent

class FieldRenderer:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.fig, self.ax = plt.subplots(figsize=(6, 10))
        self._setup_base()           # één keer bij opstart

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _setup_base(self):
        """(Re)draw the static background on a cleared axis."""
        self.ax.clear()
        self.ax.set_xlim(0, self.cfg.width)
        self.ax.set_ylim(self.cfg.height, 0)
        self.ax.set_aspect("equal")

        # veldrand
        self.ax.add_patch(
            patches.Rectangle((0, 0), self.cfg.width, self.cfg.height, fill=False, lw=2)
        )
        # notch
        self.ax.add_patch(
            patches.Rectangle(
                (self.cfg.notch_x0, self.cfg.notch_y0),
                self.cfg.notch_x1 - self.cfg.notch_x0,
                self.cfg.notch_y1 - self.cfg.notch_y0,
                color="lightgrey",
                alpha=0.3,
            )
        )

        # dynamische entiteiten (lege placeholders)
        self.robot_patch = patches.Circle((0, 0), self.cfg.robot_radius, color="blue")
        self.ax.add_patch(self.robot_patch)

        # sensor crop (lichtgeel doorschijnend vlak)
        crop = self.cfg.crop_size
        self.sensor_patch = patches.Rectangle(
            (0, 0), crop, crop,
            color="yellow",
            alpha=0.2,
            zorder=1
        )
        self.ax.add_patch(self.sensor_patch)

        self.trash_patches, self.tree_patches, self.bin_patches = [], [], []


    # aanroepen bij elke nieuwe episode
    def reset_axes(self):
        self._setup_base()

    # ------------------------------------------------------------------ #
    # initialisers en updater
    # ------------------------------------------------------------------ #
    def init_static(self, env: FestivalEnv):
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
            rect = patches.Rectangle((cx - w/2, cy - h/2), w, h,
                                    angle=deg, color="yellow", alpha=0.7)
            self.ax.add_patch(rect)

    def init_trash(self, env: FestivalEnv):
        for tx, ty in env.trash:
            circ = patches.Circle(
                (tx, ty), radius=env.cfg.trash_radius, color="red", alpha=0.8
            )
            self.ax.add_patch(circ)
            self.trash_patches.append(circ)

    def update(self, env: FestivalEnv):
        self.robot_patch.center = (env.x, env.y)
        half = self.cfg.crop_size / 2
        self.sensor_patch.set_xy((env.x - half, env.y - half))
        for circ, alive in zip(self.trash_patches, env.trash_mask):
            circ.set_visible(alive)
        plt.pause(0.001)



def run_episode(env: FestivalEnv, agent, renderer: FieldRenderer | None = None):
    obs, _ = env.reset()
    if renderer is not None:
        renderer.reset_axes()
        renderer.init_static(env)
        renderer.init_trash(env)
        renderer.update(env)
    total_reward = 0.0
    illegal_moves = 0

    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        #########################
        agent.memory[-1]["reward"] = reward
        agent.memory[-1]["done"] = done
        ###########################
        if info.get("illegal", False):
            illegal_moves += 1
        if renderer is not None:
            renderer.update(env)
    ##################
    agent.update(obs)
    return total_reward, illegal_moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = EnvConfig()
    env = FestivalEnv(cfg)
    # agent = RandomAgent(env.action_space)
    # agent = PPOAgent(env, model_path=None, total_timesteps=100_000)
    agent = PPOAgent(env.observation_space, env.action_space)

    renderer = FieldRenderer(cfg) if args.render else None

    rewards, illegals = [], []
    for _ in tqdm(range(args.episodes), desc="episodes"):
        r, ill = run_episode(env, agent, renderer)
        rewards.append(r)
        illegals.append(ill)

    env.close()
    if args.render:
        plt.show(block=True)

    print("mean reward:", np.mean(rewards))
    print("mean illegal moves per ep:", np.mean(illegals))