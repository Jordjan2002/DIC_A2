"""
Gymnasium environment for a trash‑collecting robot on a festival site.
* 37.5 m × 140 m rectangular field, with one notch on the east side.
* Static obstacles   : trees (dark‑green circles) and benches (yellow rectangles).
* Drop‑off stations : 4 bins on the left border, 4 bins on the right border.
* Robot diameter    : 1.5 m  (radius 0.75 m).
* Sensor            : 1‑channel top‑down crop 15 m × 15 m around robot, now **128×128 px**.
  pixel == 1.0 : trash            (pick‑up target)
  pixel == 0.5 : bin              (drop‑off station)
  pixel == 0.9 : wall             (obstacle)
  pixel == 0.8 : tree             (obstacle)
  pixel == 0.7 : bench            (obstacle)
  pixel == 0.0 : empty            (no trash, no bin, no tree, no bench) 

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
from numba import njit, prange
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

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
    max_trash: int = 200  # amount of trash in the field
    max_load: int = 20

    # sensor crop (↑ resolution bumped to 64x64)
    crop_size: float = 15.0
    img_px: int = 64

    # reward scalars
    step_penalty: float = -0.01
    pickup_reward: float = 1.0
    unload_reward: float = 0.5   # per item
    illegal_penalty: float = -1.0

    # episode length
    max_steps: int = 1_000

    # benches
    bench_count: int = 6
    bench_width: float = 2.0
    bench_height: float = 1.5

    # static trash and benches
    static_benches: bool = True
    static_trash: bool = True   

    # Experiment
    bench_experiment = True # Cannot be True if static_benches is False

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
    (0, 0)  # stay in place  
]


def _circle_vs_aabb(cx, cy, rad, rx, ry, w, h):
    """
    Collusion test robot (cx,cy,rad) vs. bench
    """
    dx = abs(cx - rx)
    dy = abs(cy - ry)
    if dx > (w / 2 + rad):  # too far away in x direction
        return False
    if dy > (h / 2 + rad):  # too far away in y direction
        return False
    # circle center in rectangle
    if dx <= (w / 2) or dy <= (h / 2):
        return True
    # corner distance
    return (dx - w / 2) ** 2 + (dy - h / 2) ** 2 <= rad ** 2

def _point_in_rot_rect(px, py, cx, cy, w, h, deg):
    """True if trash (px,py) within (or border) of rotated rectangle."""
    theta = math.radians(-deg)
    dx = math.cos(theta)*(px - cx) - math.sin(theta)*(py - cy)
    dy = math.sin(theta)*(px - cx) + math.cos(theta)*(py - cy)
    return abs(dx) <= w/2 and abs(dy) <= h/2


@njit(fastmath=True, cache=True)
def _sensor_fast(px, half, cell,
                 rx, ry,                       
                 trees, t_rad,
                 bins_xy,                      
                 trash_xy,
                 benches_data,
                 trash_mask,
                 robot_radius,
                 width,
                 height,
                 notch_x0,
                 notch_x1,
                 notch_y0,
                 notch_y1): 
    """ Generate the robots sensor image"""       
    img = np.zeros((px, px), dtype=np.float32)

    # -------- draws a circle --------
    def paint_disc(cx, cy, rad, value):
        gx = (cx - (rx - half)) / cell
        gy = (cy - (ry - half)) / cell
        r_px = rad / cell
        x0 = max(0, int(gx - r_px - 1))
        x1 = min(px - 1, int(gx + r_px + 1))
        y0 = max(0, int(gy - r_px - 1))
        y1 = min(px - 1, int(gy + r_px + 1))
        for iy in range(y0, y1 + 1):
            dy = iy - gy
            dy2 = dy * dy
            for ix in range(x0, x1 + 1):
                dx = ix - gx
                if dx * dx + dy2 <= r_px * r_px:
                    img[iy, ix] = value

    # draw all trees (value 0.8)
    for k in range(trees.shape[0]):
        paint_disc(trees[k, 0], trees[k, 1], t_rad, 0.8)

    # draw all bins (value 0.5)
    for k in range(bins_xy.shape[0]):
        paint_disc(bins_xy[k, 0], bins_xy[k, 1], 0.75, 0.5)

    # draw all trash (value 1.0)
    for k in range(trash_xy.shape[0]):
        if trash_mask[k]:
            paint_disc(trash_xy[k, 0], trash_xy[k, 1], 0.15, 1.0)

    # draw all benches (value 0.7)
    for k in range(benches_data.shape[0]):
        x, y, bw, bh, deg = benches_data[k]
        # paint the rectangle as a disc
        paint_disc(x, y, math.hypot(bw / 2, bh / 2), 0.7)

    near_bottom = height - ry <= 8
    near_top = ry <= 8
    near_right = width - rx <= 8
    near_left = rx <= 8
    near_notch_top = abs(notch_y1 - ry) <= 8
    near_notch_bottom = abs(notch_y0 - ry) <= 8
    near_notch_left = abs(notch_x0 - rx) <= 8

    # draw the borders (value 0.9)
    if near_bottom or near_left or near_top or near_right or near_notch_top or near_notch_bottom or near_notch_left:
        for iy in range(px):
            for ix in range(px):
                # world pixel coordinate
                gx = rx - half + ix * cell
                gy = ry - half + iy * cell

                # check if outside the field or inside the notch
                if (gx < robot_radius 
                    or gx > width  - robot_radius 
                    or gy < robot_radius 
                    or gy > height - robot_radius
                    or (notch_x0 <= gx <= notch_x1 and notch_y0 <= gy <= notch_y1)):
                    img[iy, ix] = 0.9
    return img




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

        self.create_bins()
        self.trees = [(20, 138), (1, 103.5), (4, 29), (22, 115), (30, 87), (30, 37), (22, 3)]
        self.create_trash()
        self.create_benches()
         
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
    # sensor image – now 64x64 and encodes trees & bins
    # --------------------------------------------------------------------- #

    def _sensor_image(self) -> np.ndarray:
        px   = self.cfg.img_px
        half = self.cfg.crop_size / 2
        cell = self.cfg.crop_size / px

        # lists to NumPy float32-arrays (need Numba)
        trees_arr = np.asarray(self.trees, dtype=np.float32) if len(self.trees) > 0 else np.zeros((0,2), dtype=np.float32)
        bins_arr  = np.asarray(self.bins,  dtype=np.float32) if len(self.bins)  > 0 else np.zeros((0,2), dtype=np.float32)
        trash_arr = np.asarray(self.trash, dtype=np.float32) if len(self.trash) > 0 else np.zeros((0,2), dtype=np.float32)
        benches_arr = np.asarray(self.rect_obs, dtype=np.float32) if len(self.rect_obs) > 0 else np.zeros((0,5), dtype=np.float32)
        mask_arr  = self.trash_mask.astype(np.bool_) if (self.trash_mask is not None) else np.zeros(0, dtype=np.bool_)

        # generate the sensor image
        img = _sensor_fast(px, half, cell,
                           self.x, self.y,
                           trees_arr, self.cfg.tree_radius,
                           bins_arr, trash_arr, benches_arr, mask_arr, 
                           self.cfg.robot_radius, self.cfg.width, self.cfg.height,
                           self.cfg.notch_x0, self.cfg.notch_x1,
                           self.cfg.notch_y0, self.cfg.notch_y1)
        

        return img[None, ...]

    # --------------------------------------------------------------------- #
    # Bench and Trash builders
    # --------------------------------------------------------------------- #

    def create_benches(self):
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

    def create_trash(self):
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

    def create_bins(self):
        self.bins = []
        x_left  = 2.0                     # distance to left side
        x_right = self.cfg.width - 2.0    # distance to right side

        y_positions = [20, 40, 60, 80, 100, 120] # positions on the y axis

        for y in y_positions:
            self.bins.append((x_left, y))
            self.bins.append((x_right, y))


    # --------------------------------------------------------------------- #
    # gym API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # ----------------------------------
        # 1. create yellow benches and trash if not static
        # ----------------------------------
        if self.cfg.static_benches == False:
            self.rect_obs = []
            self.create_benches() 

        if self.cfg.static_trash == False:
            self.create_trash()

        # ----------------------------------
        # 2. robot init position (centre)
        # ----------------------------------
        self.x = self.cfg.width / 2
        self.y = 35.0
        self.load = 0
        self.steps = 0

        # ----------------------------------
        # 3. randomly change the position of one bench (EXPERIMENT)
        # ----------------------------------
        if self.cfg.bench_experiment == True:
            random_index = random.randrange(len(self.rect_obs))
            popped_item = self.rect_obs.pop(random_index)
            self.cfg.bench_count = 1
            self.create_benches()


        return self._get_obs(), {}

    def _get_obs(self):
        # get the image and the state of the robot
        state_vec = np.array([
            self.x / self.cfg.width, # x position
            self.y / self.cfg.height, # y position
            self.load / self.cfg.max_load, # load percentage
            1.0 if self.load >= self.cfg.max_load else 0.0, # full yes/no
        ], dtype=np.float32)
        return {"image": self._sensor_image(), "state": state_vec}

    def step(self, action: int):
        """ Make the robot do one step """
        self.steps += 1
        reward = self.cfg.step_penalty
        terminated = truncated = False
        info = {"illegal": False}

        dx, dy = DIRECTIONS[action]

        # change positions if the robot needs to move
        if (dx, dy) != (0, 0):
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
                reward += self.load * self.cfg.unload_reward
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
        """ Print the state of the robot if required"""
        if self.render_mode != "human":
            return
        print(f"(x={self.x:.2f}, y={self.y:.2f}) load={self.load} / {self.cfg.max_load}")

    def close(self):
        pass