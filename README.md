# Festival Clean-Up Robot Environment

This repository contains a custom OpenAI Gym (Gymnasium) environment for simulating a robot that collects trash on the Flux Field of the TU/e. The robot moves around, picks up trash, and drops it off at bins when full.

The environment is built to support reinforcement learning, with a continuous state space and a discrete action space.

## Features

- Continuous 2D environment (37.5 m × 140 m) with a notch on one side.
- 8 discrete movement directions (1 meter per step).
- Robot diameter: 1.5 meters.
- Trash scattered randomly, up to 50 pieces per episode.
- 10 trash bins (5 per long side).
- Static obstacles: trees (circular) and benches (rotated rectangles).
- Robot has a 15 m × 15 m field of view, converted to a 128×128 grayscale image.
- Combined observation: sensor image + 4D state vector.
- Fully compatible with DQN, PPO, or other RL algorithms.

## State and Action Spaces


### Observation (Dict)
- `image`: shape `(1, 128, 128)`, values:
  - `1.0` → trash
  - `0.8` → trees
  - `0.7` → benches
  - `0.5` → bins
  - `0.9` → wall
- `state`: shape `(4,)`, contents:
  - `x_normalized` (0–1)
  - `y_normalized` (0–1)
  - `load_fraction` (0–1)
  - `is_full` (0 or 1)

### Action (Discrete)
- 9 directions: N, NE, E, SE, S, SW, W, NW, do nothing  
- Each step moves the robot 1 meter, except for do nothing

## Reward Function

| Situation            | Reward     |
|----------------------|------------|
| Each step            | −0.01      |
| Illegal move         | −1.00      |
| Picking up trash     | +1.00      |
| Unloading at a bin   | +5.00      |

Episodes end when:
- All trash is removed, or
- Max steps reached, or
- Robot unloads after being full.

## File Structure

project_root/
│
├── env/
│ └── festival_env.py # Custom Gym environment
│ └── stochastic_festival_env.py # Custom Gym environment with people
│
├── agents/
│ └── random_agent.py # Simple random agent
│
├── simulate.py # Runs episodes and shows matplotlib visualisation
├── requirements.txt # Dependencies
├── README.md # You are here


## Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

## Install dependencies:

pip install -r requirements.txt

## Running the Simulation
Run the simulation script with rendering:

python simulate.py --episodes 1000 --render --people-env # people walking through the environment
python simulate.py --episodes 1000 --render # no people walking through the environment

Possible arguments:
--episodes int # defines number of episodes
--render # turns on visualisation
--people-env # turns on people walking through the environment


