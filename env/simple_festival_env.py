"""
Simplified version of the festival environment for testing and debugging.
Smaller field, fewer entities, and simpler obstacles.
"""

from dataclasses import dataclass
import math
from typing import List, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np

from env.festival_trash_env import FestivalEnv, EnvConfig, DIRECTIONS


@dataclass
class SimpleEnvConfig(EnvConfig):
    # Smaller field
    width: float = 20.0
    height: float = 20.0
    
    # Fewer entities
    max_trash: int = 5
    max_load: int = 2
    
    # Simpler rewards
    step_penalty: float = -0.01
    pickup_reward: float = 2.0
    unload_reward: float = 10.0
    illegal_penalty: float = -1.0
    
    # Shorter episodes
    max_steps: int = 100


class SimpleFestivalEnv(FestivalEnv):
    def __init__(self, seed: int | None = None):
        super().__init__(cfg=SimpleEnvConfig(), seed=seed)
    
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ----------------------------------
        # 1. Static trees - just 2 trees
        # ----------------------------------
        self.trees = [
            (5, 5),    # Top-left
            (15, 15)   # Bottom-right
        ]

        # ----------------------------------
        # 2. Bins - just 2 bins
        # ----------------------------------
        self.bins = [
            (2, 10),   # Left side
            (18, 10)   # Right side
        ]

        # ----------------------------------
        # 3. Scatter trash uniformly
        # ----------------------------------
        self.trash = []
        trials = 0
        while len(self.trash) < self.cfg.max_trash and trials < self.cfg.max_trash * 20:
            x = self.rng.uniform(0, self.cfg.width)
            y = self.rng.uniform(0, self.cfg.height)
            trials += 1
            if not self._inside_field(x, y):
                continue
            if any(math.hypot(x - tx, y - ty) < self.cfg.tree_radius + self.cfg.trash_radius 
                  for tx, ty in self.trees):
                continue
            self.trash.append((x, y))
        self.trash_mask = np.ones(len(self.trash), dtype=bool)

        # ----------------------------------
        # 4. No yellow benches in simple version
        # ----------------------------------
        self.rect_obs = []

        # ----------------------------------
        # 5. Robot init position (centre)
        # ----------------------------------
        self.x = self.cfg.width / 2
        self.y = self.cfg.height / 2
        self.load = 0
        self.steps = 0

        return self._get_obs(), {} 