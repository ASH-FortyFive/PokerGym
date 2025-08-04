from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Uniform
from torch.distributions import Normal, TransformedDistribution, TanhTransform

from pokergym.rl.encoders import BackboneNetwork
from pokergym.env.enums import Action

# --- Policy Network ---
class PokerPolicy(nn.Module):
    def __init__(self, obs_space, action_space, one_hot: bool, hidden_dim=128, device=torch.device("cpu")):
        super(PokerPolicy, self).__init__()
        self.one_hot = one_hot
        self.device = device
        self.backbone = BackboneNetwork(
            observation_space=obs_space, 
            hidden_dimensions=hidden_dim,
            out_features=hidden_dim,
            one_hot=self.one_hot,
        )
        self._action_dim = action_space["action"].n 
        self.action_head = nn.Linear(hidden_dim, self._action_dim)
        self.bet_head = nn.Linear(hidden_dim, 2)  # Mean and Std for betting
        self.value_head = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, obs):
        x = self.backbone(obs)
        action_logits = self.action_head(x)
        bet_params = self.bet_head(x) 
        value = self.value_head(x)
        return action_logits, bet_params, value

    def act(self, obs, action_mask, bet_range):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
        action_logits, bet_params, value = self(obs)

        # Mask invalid actions
        action_logits = action_logits.masked_fill(~action_mask, float("-inf"))
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        bet_size = torch.tensor(0.0).to(self.device)
        bet_log_prob = torch.tensor(0.0).to(self.device)
        if action.item() == Action.RAISE.value:
            min_bet, max_bet = bet_range
            if min_bet == max_bet:
                # Deterministic case
                bet_size = torch.clone(min_bet).to(self.device)  # Start with minimum bet
                bet_log_prob = torch.tensor(0.0, device=self.device)  # log(1) = 0
            else:
                # Sample from Uniform distribution
                bet_mean, bet_std = bet_params[0], F.softplus(bet_params[1]) + 1e-6
                base_dist = Normal(bet_mean, bet_std)
                tanh_dist = TransformedDistribution(base_dist, [TanhTransform()])

                raw_sample = tanh_dist.rsample()  # in (-1, 1)
                bet_size = min_bet + 0.5 * (raw_sample + 1) * (max_bet - min_bet)  # scale to range
                bet_log_prob = tanh_dist.log_prob(raw_sample) - torch.log(0.5 * (max_bet - min_bet))

        total_log_prob = action_log_prob + bet_log_prob
        return action, bet_size, total_log_prob, entropy, value

# --- PPO Buffer ---
class PPOBuffer:
    """Buffer to store experiences for PPO training."""
    def __init__(self, device=torch.device("cpu")):
        self.obs = []
        self.actions = []  # Stores (action, bet_size) tuples
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.bet_ranges = []  # Stores (min_bet, max_bet) tuples
        self.device = device

    def store(self, obs, action, bet_size, log_prob, bet_range, value, reward):
        self.obs.append(obs)
        self.actions.append((action, bet_size))
        self.log_probs.append(log_prob)
        self.bet_ranges.append(bet_range)
        self.values.append(value)
        self.rewards.append(reward)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        self.returns = []
        self.advantages = []
        rewards = self.rewards + [last_value]
        values = self.values + [last_value]
        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            self.returns.insert(0, gae + values[t])

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get(self):
        actions, bet_sizes = zip(*self.actions)
        return (
            torch.stack(self.obs),
            torch.stack(actions),
            torch.stack(bet_sizes),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.stack(self.bet_ranges),
            self.returns,
            self.advantages
        )

    def clear(self):
        self.__init__(self.device)  # Reinitialize to clear the buffer

def ppo_update(
    policy,
    optimizer,
    data,
    clip_ratio=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    epochs=10,
):
    obs, actions, bet_sizes, old_log_probs, values, bet_ranges, returns, advantages = data
    torch.autograd.set_detect_anomaly(True)  # Keep for debugging
    for _ in range(epochs):
        action_logits, bet_params, values = policy(obs)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)

        # --- Bet log prob ---
        bet_log_probs = torch.zeros_like(log_probs)  # Initialize as zeros
        raise_mask = (actions == Action.RAISE.value)
        if raise_mask.any():
            mean = bet_params[raise_mask, 0].clone()  # Clone to avoid view issues
            std = F.softplus(bet_params[raise_mask, 1].clone()) + 1e-6  # Clone and apply softplus
            base_dist = Normal(mean, std)
            tanh_dist = TransformedDistribution(base_dist, [TanhTransform()])

            min_bet = bet_ranges[raise_mask, 0].clone()  # Clone for safety
            max_bet = bet_ranges[raise_mask, 1].clone()  # Clone for safety
            scale = 0.5 * (max_bet - min_bet)
            scaled_sample = (bet_sizes[raise_mask].clone() - min_bet) / scale - 1.0  # Clone bet_sizes

            # Compute bet log probs without in-place assignment
            bet_log_prob_values = tanh_dist.log_prob(scaled_sample) - torch.log(scale)
            bet_log_probs = bet_log_probs.scatter(0, raise_mask.nonzero(as_tuple=True)[0], bet_log_prob_values)

        total_log_probs = log_probs + bet_log_probs
        entropy = dist.entropy().mean()

        # --- PPO losses ---
        ratio = torch.exp(total_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Keep for multi-epoch updates
        optimizer.step()

    return policy_loss.item(), value_loss.item()