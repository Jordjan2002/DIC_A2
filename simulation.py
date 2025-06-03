"""
Run episodes and visualise the festival field with trees, trash, bins and the robot path.
Usage:
    python simulate.py --episodes 5 --render
"""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from tqdm import tqdm
from env import EnvConfig
from agents import RandomAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:                 # only for type checkers / IDEs
    from env import FestivalEnv
    from env.stochastic_festival_trash_env import FestivalEnvPeople


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

        # ----------------------  load indicator  ---------------------- #
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

        self.trash_patches, self.tree_patches, self.bin_patches = [], [], []
        self.person_patches = [] if self.show_people else None



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
            rect = patches.Rectangle((cx - w/2, cy - h/2), w, h,angle=deg, color="yellow", alpha=0.7, rotation_point="center")
            self.ax.add_patch(rect)

        if self.show_people and hasattr(env, "people"): #if self.show_people:
            for p in env.people:                         # may be empty at t=0
                circ = patches.Circle((p.x, p.y), 0.4, color="purple")
                self.ax.add_patch(circ)
                self.person_patches.append(circ)


    def init_trash(self, env: FestivalEnv):
        for tx, ty in env.trash:
            circ = patches.Circle(
                (tx, ty), radius=env.cfg.trash_radius, color="red", alpha=0.8
            )
            self.ax.add_patch(circ)
            self.trash_patches.append(circ)

    def update(self, env: FestivalEnv, illegal: bool = False):

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


        self.robot_patch.center = (env.x, env.y)
        half = self.cfg.crop_size / 2
        self.sensor_patch.set_xy((env.x - half, env.y - half))
        for circ, alive in zip(self.trash_patches, env.trash_mask):
            circ.set_visible(alive)

        if illegal:
            self.load_text.set_text("Collision penalty!" if illegal else "")
        else:
            self.load_text.set_text(f"Load: {env.load} / {env.cfg.max_load}")

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
        done = done or trunc  
        total_reward += reward
        if info.get("illegal", False):
            illegal_moves += 1
        if renderer is not None:
            renderer.update(env, illegal=info.get("illegal", False))
    return total_reward, illegal_moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--people-env",
        action="store_true",
        help="use FestivalEnvPeople (with moving people) instead of FestivalEnv"
    )


    args = parser.parse_args()

    # pick the environment class
    if args.people_env:
        from env.stochastic_festival_trash_env import FestivalEnvPeople as EnvClass
    else:
        from env import FestivalEnv as EnvClass

    cfg = EnvConfig(max_steps=500)
    env = EnvClass(cfg)  
    agent = RandomAgent(env.action_space)

    renderer = (FieldRenderer(cfg, show_people=args.people_env) if args.render else None)


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