"""
Gymnasium environment for a trash‑collecting robot on a festival site (TUE assignment).
* 37.5 m × 140 m rectangular field, with one notch on the east side.
* Static obstacles   : trees (dark‑green circles).
* Drop‑off stations : 4 bins on the left border, 4 bins on the right border.
* Robot diameter    : 1.5 m  (radius 0.75 m).
* Sensor            : 1‑channel top‑down crop 15 m × 15 m around robot, now **128×128 px**.
  pixel == 1.0 : trash            (pick‑up target)
  pixel == 0.5 : bin              (drop‑off station)
  pixel == 0.8 : tree (obstacle)  (optional CNN can distinguish)
  pixel == 0.9 : wall (obstacle)  (optional CNN can distinguish)
* Continuous state  : (x_norm, y_norm, load_frac, full_flag)
* Discrete actions  : 8‑way moves of 1 m per step.
* Reward
    −0.01  each step  (time penalty)
    −1.00  illegal move (wall / tree collision attempt)
    +1.00  pick‑up one trash item
    +5.00  unload (reward *per unload action*, proportional to items dropped)

The episode terminates when the robot is full and subsequently unloads **or** when all trash
is removed, or when `max_steps` is reached (truncation).
"""

from dataclasses import dataclass
import math
from typing import List, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np


# --------------------------------------------------------------------------- #
# configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    # field geometry
    width: float = 37.5
    height: float = 140.0
    notch_x0: float = 28.5
    notch_x1: float = 37.5
    notch_y0: float = 43.5
    notch_y1: float = 57.5

    # physical sizes
    robot_radius: float = 0.75
    tree_radius: float = 3.0
    trash_radius: float = 0.15

    # entities
    max_trash: int = 50
    max_load: int = 1

    # sensor crop (↑ resolution bumped to 128×128)
    crop_size: float = 15.0
    img_px: int = 128

    # reward scalars
    step_penalty: float = -0.01
    pickup_reward: float = 1.0
    unload_reward: float = 5.0   # per drop action, applied once (not per item)
    illegal_penalty: float = -1.0

    # episode length
    max_steps: int = 1000

    # benches
    bench_count: int = 6
    bench_width: float = 2.0
    bench_height: float = 1.5


# mapping action index → (dx, dy)
DIRECTIONS = [
    (0, -1),  # N
    (1, -1),  # NE
    (1, 0),   # E
    (1, 1),   # SE
    (0, 1),   # S
    (-1, 1),  # SW
    (-1, 0),  # W
    (-1, -1), # NW
]


def _circle_vs_aabb(cx, cy, rad, rx, ry, w, h):
    """
    Botsingstest cirkel (cx,cy,rad) vs. axis-aligned rectangle
    gecentreerd op (rx,ry) met breedte w en hoogte h.
    """
    dx = abs(cx - rx)
    dy = abs(cy - ry)
    if dx > (w / 2 + rad):  # too far away in x-richting
        return False
    if dy > (h / 2 + rad):  # too far away in y-richting
        return False
    # binnendoos of rand raakt
    if dx <= (w / 2) or dy <= (h / 2):
        return True
    # corner ↔ cricle
    return (dx - w / 2) ** 2 + (dy - h / 2) ** 2 <= rad ** 2

def _point_in_rot_rect(px, py, cx, cy, w, h, deg):
    """True if point (px,py) within (of op rand) of rotated rectangle."""
    theta = math.radians(-deg)
    dx = math.cos(theta)*(px - cx) - math.sin(theta)*(py - cy)
    dy = math.sin(theta)*(px - cx) + math.cos(theta)*(py - cy)
    return abs(dx) <= w/2 and abs(dy) <= h/2



class FestivalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # --------------------------------------------------------------------- #
    def __init__(self, cfg: EnvConfig = EnvConfig(), render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # observation space
        self.observation_space = Dict(
            image=Box(low=0.0, high=1.0, shape=(1, cfg.img_px, cfg.img_px), dtype=np.float32),
            state=Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
        )
        self.action_space = Discrete(len(DIRECTIONS))

        # containers defined in reset()
        self.trees: List[Tuple[float, float]] = []
        self.trash: List[Tuple[float, float]] = []
        self.trash_mask: np.ndarray | None = None
        self.bins: List[Tuple[float, float]] = []
        self.rect_obs: list[tuple[float,float,float,float,float]] = []  # (cx,cy,w,h,deg)
        self.steps = 0

    # --------------------------------------------------------------------- #
    # geometry helpers
    # --------------------------------------------------------------------- #
    def _inside_field(self, x: float, y: float) -> bool:
        # robot centre must stay >= robot_radius from outer walls
        if not (self.cfg.robot_radius <= x <= self.cfg.width - self.cfg.robot_radius and
                self.cfg.robot_radius <= y <= self.cfg.height - self.cfg.robot_radius):
            return False
        # check notch (no radius offset, centre only)
        in_notch_x = self.cfg.notch_x0 < x < self.cfg.notch_x1
        in_notch_y = self.cfg.notch_y0 < y < self.cfg.notch_y1
        return not (in_notch_x and in_notch_y)

    def _collides(self, x: float, y: float) -> bool:
        if not self._inside_field(x, y):
            return True
        # tree collisions
        for tx, ty in self.trees:
            if math.hypot(x - tx, y - ty) < self.cfg.tree_radius + self.cfg.robot_radius:
                return True
            
        # yellow bench collisions
        for bx, by, bw, bh, _ in self.rect_obs:          # corner 0° → '_'
            if _circle_vs_aabb(x, y, self.cfg.robot_radius, bx, by, bw, bh):
                return True

        return False

    # --------------------------------------------------------------------- #
    # sensor image – now 128×128 and encodes trees & bins
    # --------------------------------------------------------------------- #
    def _sensor_image(self) -> np.ndarray:
        px = self.cfg.img_px
        half = self.cfg.crop_size / 2
        cell = self.cfg.crop_size / px
        img = np.zeros((px, px), dtype=np.float32)

        # helper to paint discs in grid coords
        def _paint_circle(cx, cy, rad, value):
            gx = int((cx - (self.x - half)) / cell)
            gy = int((cy - (self.y - half)) / cell)
            r_px = int(rad / cell) + 1
            for dy in range(-r_px, r_px + 1):
                for dx in range(-r_px, r_px + 1):
                    ix = gx + dx
                    iy = gy + dy
                    if 0 <= ix < px and 0 <= iy < px:
                        if dx * dx + dy * dy <= r_px * r_px:
                            img[iy, ix] = value

        # trees  (value 0.8)
        for tx, ty in self.trees:
            if abs(tx - self.x) <= half + self.cfg.tree_radius and abs(ty - self.y) <= half + self.cfg.tree_radius:
                _paint_circle(tx, ty, self.cfg.tree_radius, 0.8)
        # bins   (value 0.5)
        bin_r = 0.75
        for bx, by in self.bins:
            x0 = int((bx - bin_r - (self.x - half)) / cell)
            x1 = int((bx + bin_r - (self.x - half)) / cell)
            y0 = int((by - bin_r - (self.y - half)) / cell)
            y1 = int((by + bin_r - (self.y - half)) / cell)
            x0, x1 = max(0, x0), min(px - 1, x1)
            y0, y1 = max(0, y0), min(px - 1, y1)
            img[y0:y1+1, x0:x1+1] = 0.5

        # trash  (value 1.0)
        for (tx, ty), alive in zip(self.trash, self.trash_mask):
            if not alive:
                continue
            if abs(tx - self.x) <= half and abs(ty - self.y) <= half:
                _paint_circle(tx, ty, self.cfg.trash_radius, 1.0)

        # rechthoeken  (waarde 0.6)
        for rx, ry, w, h, deg in self.rect_obs:
            # simpele rastering: bounding box in grid-coord
            ang = math.radians(deg)
            # corners in worldcoordinates
            corners = []
            for sx in (-w/2, w/2):
                for sy in (-h/2, h/2):
                    cx = rx + math.cos(ang)*sx - math.sin(ang)*sy
                    cy = ry + math.sin(ang)*sx + math.cos(ang)*sy
                    corners.append((cx, cy))
            xs, ys = zip(*corners)
            if (max(xs) < self.x - half or min(xs) > self.x + half or
                max(ys) < self.y - half or min(ys) > self.y + half):
                continue
            # paint rough bbox in grid (fast, not pixel-perfect)
            _paint_circle(rx, ry, max(w, h)/2, 0.6)

        # walls → value 0.9
        cell = self.cfg.crop_size / self.cfg.img_px
        half = self.cfg.crop_size / 2
        for i in range(self.cfg.img_px):
            for j in range(self.cfg.img_px):
                gx = self.x - half + j * cell
                gy = self.y - half + i * cell
                if (gx < self.cfg.robot_radius or gx > self.cfg.width - self.cfg.robot_radius or
                    gy < self.cfg.robot_radius or gy > self.cfg.height - self.cfg.robot_radius):
                    img[i, j] = 0.9


        return img[None, ...]

    # --------------------------------------------------------------------- #
    # gym API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ----------------------------------
        # 1. static trees – rough positions from aerial photo (manually measured)
        #    coordinates are approximate in metres from NW origin.
        # ----------------------------------
        self.trees = [(20, 138), (1, 103.5), (4, 29), (22, 115), (30, 87), (30, 37), (22, 3)]
        # ----------------------------------
        # 2. bins – four per long side, spaced evenly, flush to the wall
        # ----------------------------------
        # gap_y = self.cfg.height / 6
        # self.bins = []
        # for i in range(1, 5):
        #     y_pos = i * gap_y
        #     self.bins.append((self.cfg.robot_radius, y_pos))               # left side
        #     self.bins.append((self.cfg.width - self.cfg.robot_radius, y_pos))  # right side

        self.bins = []
        x_left  = 2.0                      # afstand tot linkerzijde
        x_right = self.cfg.width - 2.0    # afstand tot rechterzijde

        y_positions = [20, 40, 60, 80, 100, 120]  # iets omhooggeschoven + extra onderaan

        for y in y_positions:
            self.bins.append((x_left, y))
            self.bins.append((x_right, y))

        # ----------------------------------
        # 3. scatter trash uniformly
        # ----------------------------------
        self.trash = []
        trials = 0
        while len(self.trash) < self.cfg.max_trash and trials < self.cfg.max_trash * 20:
            x = self.rng.uniform(0, self.cfg.width)
            y = self.rng.uniform(0, self.cfg.height)
            trials += 1
            if not self._inside_field(x, y):
                continue
            if any(math.hypot(x - tx, y - ty) < self.cfg.tree_radius + self.cfg.trash_radius for tx, ty in self.trees):
                continue
            if any(_point_in_rot_rect( x, y, bx, by, bw, bh, deg )
                    for bx, by, bw, bh, deg in self.rect_obs):
                continue 
            self.trash.append((x, y))
        self.trash_mask = np.ones(len(self.trash), dtype=bool)

        # ----------------------------------
        # 4. create yellow benches
        # ----------------------------------
        self.rect_obs = []

        attempts = 0
        max_attempts = 1000
        benches_created = 0

        while benches_created < self.cfg.bench_count and attempts < max_attempts:
            attempts += 1
            # Random position inside the field boundaries (keeping robot radius margin)
            x = self.rng.uniform(self.cfg.robot_radius, self.cfg.width - self.cfg.robot_radius)
            y = self.rng.uniform(self.cfg.robot_radius, self.cfg.height - self.cfg.robot_radius)

            # Random rotation angle [0, 360)
            deg = self.rng.uniform(0, 360)

            # Check collisions with trees or bins
            collides = False

            # Check collision with trees
            for tx, ty in self.trees:
                # Approximate collision check: distance between bench center and tree less than sum of radii plus some margin
                # Here we simplify by using distance between centers and bench max half-diagonal
                bench_radius = math.hypot(self.cfg.bench_width / 2, self.cfg.bench_height / 2)
                if math.hypot(x - tx, y - ty) < bench_radius + self.cfg.tree_radius:
                    collides = True
                    break

            if collides:
                continue

            # Check collision with bins
            bin_radius = 0.75
            for bx, by in self.bins:
                bench_radius = math.hypot(self.cfg.bench_width / 2, self.cfg.bench_height / 2)
                if math.hypot(x - bx, y - by) < bench_radius + bin_radius:
                    collides = True
                    break

            if collides:
                continue

            # If no collision, add bench
            self.rect_obs.append((x, y, self.cfg.bench_width, self.cfg.bench_height, deg))
            benches_created += 1

        # Remove trash underneath benches
        new_trash = []
        new_mask = []
        for (tx, ty), alive in zip(self.trash, self.trash_mask):
            if not alive:
                continue
            # Check if trash is under any bench (using _point_in_rot_rect helper)
            under_bench = False
            for bx, by, bw, bh, bdeg in self.rect_obs:
                if _point_in_rot_rect(tx, ty, bx, by, bw, bh, bdeg):
                    under_bench = True
                    break
            if not under_bench:
                new_trash.append((tx, ty))
                new_mask.append(True)
        self.trash = new_trash
        self.trash_mask = np.array(new_mask, dtype=bool)

        # ----------------------------------
        # 5. robot init position (centre)
        # ----------------------------------
        self.x = self.cfg.width / 2
        self.y = 35.0
        self.load = 0
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        state_vec = np.array([
            self.x / self.cfg.width,
            self.y / self.cfg.height,
            self.load / self.cfg.max_load,
            1.0 if self.load >= self.cfg.max_load else 0.0,
        ], dtype=np.float32)
        return {"image": self._sensor_image(), "state": state_vec}

    def step(self, action: int):
        self.steps += 1
        reward = self.cfg.step_penalty
        terminated = truncated = False
        info = {"illegal": False}

        dx, dy = DIRECTIONS[action]
        nx = self.x + dx * 1.0
        ny = self.y + dy * 1.0

        # illegal move handling
        if self._collides(nx, ny):
            reward += self.cfg.illegal_penalty
            info["illegal"] = True
        else:
            self.x, self.y = nx, ny

        # drop‑off: if at a bin (< 1 m)
        for bx, by in self.bins:
            if math.hypot(self.x - bx, self.y - by) <= 1.0 and self.load > 0:
                reward += self.cfg.unload_reward
                self.load = 0
                break

        # pick‑up (only when not full)
        if self.load < self.cfg.max_load:
            for idx, (tx, ty) in enumerate(self.trash):
                if self.trash_mask[idx] and math.hypot(self.x - tx, self.y - ty) <= 1.0:
                    self.trash_mask[idx] = False
                    self.load += 1
                    reward += self.cfg.pickup_reward
                    break

        # terminating conditions
        if not self.trash_mask.any():
            terminated = True
        elif self.steps >= self.cfg.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # minimal render (text or matplotlib handled by simulate.py)
    # --------------------------------------------------------------------- #
    def render(self):
        if self.render_mode != "human":
            return
        print(f"(x={self.x:.2f}, y={self.y:.2f}) load={self.load} / {self.cfg.max_load}")

    def close(self):
        pass