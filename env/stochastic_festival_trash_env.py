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
  pixel == 0.3 : people           (moving obstacle)
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


import math
import numpy as np
from numba import njit

from .festival_trash_env import EnvConfig, DIRECTIONS, _circle_vs_aabb, FestivalEnv

class Person:
    """Moving festival visitor."""
    SPEED = 0.9         # m per step (same as robot)
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx    # unit vector already scaled to SPEED
        self.vy = vy

    def step(self):
        self.x += self.vx
        self.y += self.vy

@njit(fastmath=True, cache=True)
def _sensor_fast(px, half, cell,
                 rx, ry,                       
                 trees, t_rad,
                 bins_xy,                      
                 trash_xy,
                 benches_data,
                 people_xyvxvy,
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
        x, y, bw, bh, _ = benches_data[k]
        # paint the rectangle as a disc
        paint_disc(x, y, math.hypot(bw / 2, bh / 2), 0.7)

    # draw all people (value 0.3)
    for k in range(people_xyvxvy.shape[0]):
        x, y, _, _ = people_xyvxvy[k]
        paint_disc(x, y, 0.4, 0.3)

    # draw the borders (value 0.9)
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
    print(img)
    return img


class FestivalEnvPeople(FestivalEnv):
    """ Enhanced FestivalEnv such that it supports people moving through the field."""
    def __init__(self, cfg: EnvConfig = EnvConfig(), render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.people: list[Person] = []
        self.person_rng = np.random.default_rng(seed)
        self.person_arrival_rate = 0.4   # (expected arrivals per step)

    def _collides(self, x: float, y: float) -> bool:
        """ Check if the robot collides with obstacles or people """
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
            
        # people collisions
        for p in getattr(self, "people", []):
            if math.hypot(x - p.x, y - p.y) <= self.cfg.robot_radius + 0.4:
                return True

        return False
    
    def _people_blocked(self, x: float, y: float) -> bool:
        """Return True if Person (x,y) hits the robot or another visitor."""
        # hit the robot
        if math.hypot(x - self.x, y - self.y) <= 0.75 + 0.4:       # robot r + person r
            return True
        # hit another visitor
        return any(math.hypot(x - q.x, y - q.y) < 0.8 for q in self.people)
    
    def _sensor_image(self) -> np.ndarray:
        px   = self.cfg.img_px
        half = self.cfg.crop_size / 2
        cell = self.cfg.crop_size / px

        # lists to NumPy float32-arrays (need Numba)
        trees_arr = np.asarray(self.trees, dtype=np.float32) if len(self.trees) > 0 else np.zeros((0,2), dtype=np.float32)
        bins_arr  = np.asarray(self.bins,  dtype=np.float32) if len(self.bins)  > 0 else np.zeros((0,2), dtype=np.float32)
        trash_arr = np.asarray(self.trash, dtype=np.float32) if len(self.trash) > 0 else np.zeros((0,2), dtype=np.float32)
        benches_arr = np.asarray(self.rect_obs, dtype=np.float32) if len(self.rect_obs) > 0 else np.zeros((0,5), dtype=np.float32)
        people_arr = (np.array([[p.x, p.y, p.vx, p.vy] for p in self.people], dtype=np.float32) if len(self.people) > 0 else np.zeros((0, 4), dtype=np.float32)
)
        mask_arr  = self.trash_mask.astype(np.bool_) if (self.trash_mask is not None) else np.zeros(0, dtype=np.bool_)
         

        # generate the sensor image
        img = _sensor_fast(px, half, cell,
                           self.x, self.y,
                           trees_arr, self.cfg.tree_radius,
                           bins_arr, trash_arr, benches_arr, people_arr, mask_arr,
                           self.cfg.robot_radius, self.cfg.width, self.cfg.height,
                           self.cfg.notch_x0, self.cfg.notch_x1,
                           self.cfg.notch_y0, self.cfg.notch_y1)
        

        return img[None, ...]    

    def step(self, action: int):
        """ Let the robot and people take one step and """  
        self.steps += 1

        # insert new people into the field using a poisson distribution
        n_new = self.person_rng.poisson(self.person_arrival_rate)

        for _ in range(n_new):
            for _ in range(15):                       # <= retry up to 15 times
                edge = self.person_rng.choice(("top", "bottom", "left", "right")) # Choose which side the person comes from
                if edge == "top":
                    x, y        = self.person_rng.uniform(0, self.cfg.width), 0.0
                    dir_choices = [(0, 1), (-1, 1), (1, 1)]
                elif edge == "bottom":
                    x, y        = self.person_rng.uniform(0, self.cfg.width), self.cfg.height
                    dir_choices = [(0, -1), (-1, -1), (1, -1)]
                elif edge == "left":
                    x, y        = 0.0, self.person_rng.uniform(0, self.cfg.height)
                    dir_choices = [(1, 0), (1, -1), (1, 1)]
                else:  # right
                    x, y        = self.cfg.width, self.person_rng.uniform(0, self.cfg.height)
                    dir_choices = [(-1, 0), (-1, -1), (-1, 1)]

                self.person_rng.shuffle(dir_choices)

                for vx, vy in dir_choices:            # test each of the 3 headings
                    nx, ny = x + vx, y + vy           # first step into the field
                    # must be inside and collision-free — incl. no other visitor
                    if (self._inside_field(nx, ny) and
                        not self._collides(nx, ny) and
                        not any(math.hypot(nx - p.x, ny - p.y) < 0.8 for p in self.people)):
                        self.people.append(Person(x, y, vx, vy))
                        break                         # person spawned, exit dir loop
                else:
                    continue      # none of the 3 dirs worked → try another edge/pos
                break             # spawned → stop retry loop for this person  


        # ----------------  move existing people  ----------------
        alive: list[Person] = []

        for p in self.people:
            # # 1 candidate directions based on current heading
            # if   p.vx > 0:  dirs = [(1, 0), (1, 1), (1, -1)]          # east
            # elif p.vx < 0:  dirs = [(-1, 0), (-1, 1), (-1, -1)]       # west
            # elif p.vy > 0:  dirs = [(0, 1), (1, 1), (-1, 1)]          # south
            # else:            dirs = [(0,-1), (1,-1), (-1,-1)]         # north

            if   p.vy > 0:                      # moves to the south
                dirs = [(0, 1), (1, 1), (-1, 1)]
            elif p.vy < 0:                      # moves to the north
                dirs = [(0,-1), (1,-1), (-1,-1)]
            elif p.vx > 0:                      # moves to the east
                dirs = [(1, 0), (1, 1), (1,-1)]
            elif p.vx < 0:                      # moves to the west 
                dirs = [(-1, 0), (-1, 1), (-1,-1)]
            else:                               # safety implementation (should not happen)
                dirs = [(0,1), (0,-1), (1,0), (-1,0)]

            self.person_rng.shuffle(dirs)
            moved = False                 # <- will become True if a legal step found

            for vx, vy in dirs:
                nx, ny = p.x + vx * Person.SPEED, p.y + vy * Person.SPEED

                # 2 hits wall --> visitor immediately leaves (do not append)
                if nx < 0 or nx > self.cfg.width or ny < 0 or ny > self.cfg.height:
                    break

                # 3 hits bin --> visitor also leaves
                if any(math.hypot(nx - bx, ny - by) <= 0.75 + 0.4 for bx, by in self.bins):
                    break

                # 4 hits static obstacles / robot / other visitors
                if self._people_blocked(nx, ny):
                    continue
                if any(math.hypot(nx - q.x, ny - q.y) < 0.8 for q in self.people if q is not p):
                    continue

                # 5 free step – perform it
                p.x, p.y, p.vx, p.vy = nx, ny, vx, vy
                moved = True
                break

            if moved:                      # keep only after a successful inside-step
                alive.append(p)

        self.people = alive

        # initialize reward and termination/truncation
        reward = self.cfg.step_penalty
        terminated = truncated = False
        info = {"illegal": False}

        dx, dy = DIRECTIONS[action]

        # move the robot
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

