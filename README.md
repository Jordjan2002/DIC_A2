# Festival Clean-Up Robot Environment

This repository contains a custom OpenAI Gym (Gymnasium) environment for simulating a robot that collects trash on the Flux Field of the TU/e. The robot moves around, picks up trash, and drops it off at bins when full.

The environment supports reinforcement learning with both DQN and PPO agents, and includes a comprehensive experiment system for evaluating agent performance across different environment configurations.

## Features

- Continuous 2D environment (37.5 m × 140 m) with a notch on one side.
- 8 discrete movement directions (1 meter per step).
- Robot diameter: 1.5 meters.
- Trash scattered randomly, up to 50 pieces per episode.
- 10 trash bins (5 per long side).
- Static/dynamic obstacles: trees (circular) and benches (rotated rectangles).
- Robot has a 15 m × 15 m field of view, converted to a 64×64 image.
- Combined observation: sensor image + 4D state vector.
- Fully compatible with DQN, PPO, or other RL algorithms.

## State and Action Spaces

### Observation (Dict)
- `image`: shape `(4, 64, 64)`, values:
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
- 8 directions: N, NE, E, SE, S, SW, W, NW
- Each step moves the robot 1 meter

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

## Trained Models

The repository includes pre-trained models in the `final_models/` directory:

- **DQN Agent**: `dqn_model_2025-06-25__14-20-53.pt` - Deep Q-Network with experience replay
- **PPO Agent**: `ppo_agent1.pt` - Proximal Policy Optimization agent

## Experiment System

The comprehensive experiment system evaluates both agents across multiple environment configurations:

### Environment Variants Tested:
1. **Normal** - Static trash and benches
2. **Stochastic Benches** - Moving benches, static trash  
3. **Stochastic Trash** - Moving trash, static benches
4. **Fully Stochastic** - Both trash and benches moving

### Metrics Tracked:
- **Mean Reward** per environment setting
- **Standard Deviation** of rewards (variability)
- **Illegal Moves** (collision penalties)
- **Execution Time** per test

### Outputs Generated:
- **4 Different Visualizations**: Bar charts, heatmaps, difficulty ranking, performance profiles
- **Multiple File Formats**: JSON (raw data), CSV (summary), PNG (plots), TXT (reports)

## Usage Commands

### Quick Experiment Testing
```bash
# Ultra-minimal test (2 episodes, 10 steps each)
python test_experiments.py

# Quick comprehensive test (10 episodes, 50 steps each)
python final_experiment.py --quick

# Custom minimal run (5 episodes, 30 steps each)
python final_experiment.py --episodes 5 --max_steps 30
```

### Individual Agent Evaluation
```bash
# Test DQN agent only
python simple_evaluate.py --model_path final_models/dqn_model_2025-06-25__14-20-53.pt --episodes 3

# Test PPO agent only
python simple_evaluate_ppo.py --model_path final_models/ppo_agent1.pt --episodes 3 --env_type normal
```

### Full Experiment Suite
```bash
# Complete evaluation (50 episodes, 100 steps each) - WARNING: Memory intensive
python final_experiment.py

# Save results to custom directory
python final_experiment.py --quick --save_dir my_results
```

### Visual Simulation
```bash
# Watch DQN agent perform (with GUI)
python simulation.py --agent dqn --model final_models/dqn_model_2025-06-25__14-20-53.pt --episodes 3 --render

# Watch PPO agent perform (with GUI)
python simulation.py --agent ppo --model final_models/ppo_agent1.pt --episodes 3 --render

# Test without GUI (faster)
python simulation.py --agent dqn --model final_models/dqn_model_2025-06-25__14-20-53.pt --episodes 3
```

## File Structure

```
project_root/
│
├── env/
│   ├── festival_trash_env.py          # Main environment
│   └── simple_festival_env.py         # Simplified version
│
├── agents/
│   ├── base_agent.py                  # Base agent class
│   ├── dqn_agent.py                   # Deep Q-Network agent
│   ├── ppo_agent.py                   # PPO agent
│   └── random_agent.py               # Random baseline
│
├── final_models/
│   ├── dqn_model_2025-06-25__14-20-53.pt  # Trained DQN
│   └── ppo_agent1.pt                 # Trained PPO
│
├── experiment_results/                # Generated experiment outputs
│
├── final_experiment.py               # Comprehensive experiment runner
├── simple_evaluate.py               # Basic DQN evaluation
├── simple_evaluate_ppo.py           # PPO evaluation with compatibility
├── test_experiments.py              # Quick testing script
├── simulation.py                    # Visual simulation
├── train_dqn.py                     # DQN training script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Memory Management

If you encounter memory issues:

1. **Start with minimal tests**: `python test_experiments.py`
2. **Use quick mode**: `python final_experiment.py --quick`
3. **Set memory limits**: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python final_experiment.py --episodes 3`
4. **Test individual agents** before running full experiments

## Results Analysis

The experiment system generates detailed analysis including:

- **Performance comparison** between DQN and PPO agents
- **Environment difficulty ranking** based on average performance
- **Statistical significance** testing with error bars
- **Execution time analysis** for performance optimization
- **Comprehensive reports** in multiple formats for further analysis


