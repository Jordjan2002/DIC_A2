import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 2.5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, img_shape, state_dim, n_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_input = torch.zeros(1, *img_shape)
        n_cnn_features = self.cnn(dummy_input).shape[1]

        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(n_cnn_features + 64, 256),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, img, state):
        img_features = self.cnn(img)
        state_features = self.state_fc(state)
        combined = torch.cat([img_features, state_features], dim=1)
        x = self.fc(combined)
        return self.policy_head(x), self.value_head(x)


class PPOAgent:
    def __init__(self, observation_space, action_space):
        img_shape = observation_space["image"].shape
        state_dim = observation_space["state"].shape[0]
        self.n_actions = action_space.n

        self.model = ActorCritic(img_shape, state_dim, self.n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.memory = []

    def act(self, obs):
        img = torch.from_numpy(obs["image"]).float().to(DEVICE).unsqueeze(0)
        state = torch.from_numpy(obs["state"]).float().to(DEVICE).unsqueeze(0)

        logits, value = self.model(img, state)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.memory.append({
            "obs": obs,
            "action": action.item(),
            "log_prob": dist.log_prob(action).item(),
            "value": value.item()
        })

        return action.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0.0]  # bootstrap
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + GAMMA * LAMBDA * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, last_obs):
        # Prepare data
        obs_imgs = torch.from_numpy(np.array([r["obs"]["image"] for r in self.memory])).float().to(DEVICE)
        obs_states = torch.from_numpy(np.array([r["obs"]["state"] for r in self.memory])).float().to(DEVICE)
        actions = torch.tensor([r["action"] for r in self.memory]).to(DEVICE)
        old_log_probs = torch.tensor([r["log_prob"] for r in self.memory]).to(DEVICE)
        rewards = [r.get("reward", 0.0) for r in self.memory]
        dones = [r.get("done", False) for r in self.memory]
        values = [r["value"] for r in self.memory]

        # Compute last value for bootstrap
        img = torch.from_numpy(last_obs["image"]).float().to(DEVICE).unsqueeze(0)
        state = torch.from_numpy(last_obs["state"]).float().to(DEVICE).unsqueeze(0)
        _, last_value = self.model(img, state)
        last_value = last_value.item()

        # Compute GAE and returns
        values += [last_value]
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Convert to tensors
        returns = torch.tensor(returns).float().to(DEVICE)
        advantages = torch.tensor(advantages).float().to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(4):  # num epochs
            idx = np.arange(len(self.memory))
            np.random.shuffle(idx)
            for start in range(0, len(self.memory), 64):
                end = start + 64
                batch_idx = idx[start:end]

                b_img = obs_imgs[batch_idx]
                b_state = obs_states[batch_idx]
                b_action = actions[batch_idx]
                b_old_logp = old_log_probs[batch_idx]
                b_ret = returns[batch_idx]
                b_adv = advantages[batch_idx]

                logits, values_pred = self.model(b_img, b_state)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(b_action)
                entropy = dist.entropy().mean()

                ratio = (log_probs - b_old_logp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values_pred.squeeze(), b_ret)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear memory
        self.memory = []
