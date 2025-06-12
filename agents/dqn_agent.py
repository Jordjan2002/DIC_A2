"""
DQN agent for the festival environment.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pathlib import Path
from agents.base_agent import BaseAgent


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super(DQN, self).__init__()
        
        # Process image input (4x64x64) - efficient conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=8, stride=4),    # 4 channels -> 8 features, 64x64 -> 15x15
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),   # 8 -> 16 features, 15x15 -> 6x6
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size of flattened conv output
        # Input: 4x64x64
        # After conv1 (8x8, stride 4): 8x15x15 (since (64-8)/4 + 1 = 15)
        # After conv2 (4x4, stride 2): 16x6x6 (since (15-4)/2 + 1 = 6)
        # Flattened: 16 * 6 * 6 = 576
        self.conv_output_size = 576
        
        # Process state vector (4 values)
        self.state_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU()
        )
        
        # Combined network
        self.combined = nn.Sequential(
            nn.Linear(self.conv_output_size + 8, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
    def forward(self, image, state):
        # Process image
        conv_out = self.conv(image)
        
        # Process state
        state_out = self.state_net(state)
        
        # Combine features
        combined = torch.cat([conv_out, state_out], dim=1)
        
        # Get Q-values
        return self.combined(combined)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int = 8,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 128,
        target_update: int = 500,
        normalize_rewards: bool = True,
        seed: int | None = None
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.normalize_rewards = normalize_rewards
        
        # Reward normalization statistics
        self.reward_sum = 0.0
        self.reward_sum_squared = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_actions).to(self.device)
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        
        # Initialize memory
        self.memory = deque(maxlen=memory_size)
        
        # Training tracking
        self.steps = 0
        self.episode_rewards = []
        self.last_episode_reward = 0
        self.losses = []  # Track losses
        self.q_values = []  # Track Q-values

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        if not self.normalize_rewards:
            return reward
            
        # Update running statistics
        self.reward_count += 1
        self.reward_sum += reward
        self.reward_sum_squared += reward * reward
        
        # Calculate running mean and std
        self.reward_mean = self.reward_sum / self.reward_count
        
        if self.reward_count > 1:
            variance = (self.reward_sum_squared / self.reward_count) - (self.reward_mean ** 2)
            self.reward_std = max(np.sqrt(variance), 1e-8)  # Avoid division by zero
        
        # Normalize the reward
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        
        return normalized_reward

    def act(self, obs: dict) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon: #Explore
            return random.randrange(self.n_actions)
        else: #Exploit
            with torch.no_grad():
                # Convert observation to tensors
                image = torch.FloatTensor(obs["image"]).unsqueeze(0).to(self.device)
                state = torch.FloatTensor(obs["state"]).unsqueeze(0).to(self.device)
                
                # Get Q-values
                q_values = self.policy_net(image, state)
                
                # Return best action
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, warmup=False):
        # Normalize reward
        normalized_reward = self._normalize_reward(reward)
        
        self.memory.append((state, action, normalized_reward, next_state))
        self.last_episode_reward += reward  # Keep original reward for tracking
        self.steps += 1
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size and not warmup:
            self._train_on_batch()
            
            # Update target network
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_on_batch(self):
        """Train on a batch of experiences."""
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        images = torch.FloatTensor(np.array([x[0]["image"] for x in batch])).to(self.device)
        states = torch.FloatTensor(np.array([x[0]["state"] for x in batch])).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        
        # Get current Q-values
        current_q_values = self.policy_net(images, states).gather(1, actions.unsqueeze(1))
        # print(f"Current Q-values: {current_q_values}")
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(images, states).max(1)[0]
            # print(f"Next Q-values: {next_q_values}")
            # Compute target Q-values
            target_q_values = rewards + self.gamma * next_q_values
            # print(f"Target Q-values: {target_q_values}")
        

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Track metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())


    def end_episode(self):
        """End of episode processing."""
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Track episode reward
        self.episode_rewards.append(self.last_episode_reward)
        self.last_episode_reward = 0

    def save(self, path: Path):
        """Save the model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'q_values': self.q_values,
            'reward_sum': self.reward_sum,
            'reward_sum_squared': self.reward_sum_squared,
            'reward_count': self.reward_count,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }, path)

    def load(self, path: Path):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        self.q_values = checkpoint['q_values']
        
        # Load reward normalization statistics if available
        if 'reward_sum' in checkpoint:
            self.reward_sum = checkpoint['reward_sum']
            self.reward_sum_squared = checkpoint['reward_sum_squared']
            self.reward_count = checkpoint['reward_count']
            self.reward_mean = checkpoint['reward_mean']
            self.reward_std = checkpoint['reward_std'] 