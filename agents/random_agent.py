import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return self.action_space.sample()