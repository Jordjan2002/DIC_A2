import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict
import random
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, img_channels: int, img_size: int, state_dim: int, hidden_dim: int, output_dim: int):
        super(ActorCritic, self).__init__()
        
        # CNN for processing image (unchanged)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples from 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples from 32x32 -> 16x16
            nn.Flatten()
        )
        
        cnn_output_size = 32 * (img_size // 4) * (img_size // 4)
        
        # Separate processing pathways
        self.image_processor = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined feature processor
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: Dict[str, torch.Tensor]):
        # Process image through CNN
        img_features = self.cnn(obs['image'])  # Change to (batch_size, channels, height, width)
        
        # Process image features through dedicated pathway
        img_processed = self.image_processor(img_features)
        
        # Process state vector through separate pathway
        state_processed = self.state_processor(obs['state'])
        
        # Combine using element-wise addition
        combined = img_processed + state_processed
        
        # Further process combined features
        features = self.combined_processor(combined)
        
        # Get action probabilities and state value
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value

class PPOAgent:
    def __init__(
        self,
        action_space,
        img_channels: int = 1,
        img_size: int = 64,
        state_dim: int = 4,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.015,
        train_iters: int = 10,
        batch_size: int = 128,
        seed: int = None
    ):
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(
            img_channels=img_channels,
            img_size=img_size,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=action_space.n
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_iters = train_iters
        self.batch_size = batch_size
        
        # Experience buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.done_buffer = []
        
    def act(self, obs: Dict[str, np.ndarray]):
        with torch.no_grad():
            # Convert observations to tensors
            obs_tensor = {
                'image': torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device),
                'state': torch.FloatTensor(obs['state']).unsqueeze(0).to(self.device)
            }


            
            action_probs, value = self.actor_critic(obs_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return advantages, returns
    
    def load(self, model_file: str):
        checkpoint = torch.load(model_file, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.actor_critic.to(self.device)
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.done_buffer = []
        return self
    
    def update(self):
        # Convert buffers to tensors
        obs_tensor = {
            'image': torch.FloatTensor(np.array([o['image'] for o in self.obs_buffer])).to(self.device),
            'state': torch.FloatTensor(np.array([o['state'] for o in self.obs_buffer])).to(self.device)
        }
        action_tensor = torch.LongTensor(self.action_buffer).to(self.device)
        old_value_tensor = torch.FloatTensor(self.value_buffer).to(self.device)
        old_log_prob_tensor = torch.FloatTensor(self.log_prob_buffer).to(self.device)
        
        # Compute GAE
        with torch.no_grad():
            _, next_value = self.actor_critic({
                'image': obs_tensor['image'][-1:],
                'state': obs_tensor['state'][-1:]
            })
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(
            self.reward_buffer,
            self.value_buffer,
            self.done_buffer,
            next_value
        )
        
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        for _ in range(self.train_iters):
            # Create mini-batches
            indices = np.random.permutation(len(self.obs_buffer))
            for start_idx in range(0, len(self.obs_buffer), self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Get mini-batch data
                obs_batch = {
                    'image': obs_tensor['image'][idx],
                    'state': obs_tensor['state'][idx]
                }
                action_batch = action_tensor[idx]
                old_value_batch = old_value_tensor[idx]
                old_log_prob_batch = old_log_prob_tensor[idx]
                advantages_batch = advantages_tensor[idx]
                returns_batch = returns_tensor[idx]
                
                # Forward pass
                action_probs, values = self.actor_critic(obs_batch)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()
                
                # Calculate ratios
                ratio = torch.exp(log_probs - old_log_prob_batch)
                
                # Calculate losses
                policy_loss1 = -advantages_batch * ratio
                policy_loss2 = -advantages_batch * torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss with clipping
                value_pred_clipped = old_value_batch + (values.squeeze() - old_value_batch).clamp(-self.clip_ratio, self.clip_ratio)
                value_loss1 = (returns_batch - values.squeeze()).pow(2)
                value_loss2 = (returns_batch - value_pred_clipped).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Total loss
                loss = policy_loss + value_loss - 0.01 * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.done_buffer = []
    
    def store_transition(self, obs: Dict[str, np.ndarray], action: int, reward: float, value: float, log_prob: float, done: bool):
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.log_prob_buffer.append(log_prob)
        self.done_buffer.append(done)