from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Categorical,
    Normal,
    TanhTransform,
    TransformedDistribution,
    Uniform,
)

from pokergym.env.enums import Action
from pokergym.rl.encoders import BackboneNetwork


# --- Policy Network ---
class PokerPolicy(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        one_hot: bool,
        hidden_dim=128,
        device=torch.device("cpu"),
    ):
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
        """Process observation through the backbone and heads.
        Args:
            obs (torch.Tensor): Input observation tensor.
        Returns:
            action_logits (torch.Tensor): Logits for action selection.
            bet_params (torch.Tensor): Parameters for betting (mean and std).
            value (torch.Tensor): Value estimate for the state.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension if needed
        x = self.backbone(obs)
        action_logits = self.action_head(x)  # [batch_size, action_dim]
        bet_params = self.bet_head(x)  # [batch_size, 2]
        value = self.value_head(x)  # [batch_size, 1] logits. Check input data.")
        return action_logits, bet_params, value

    def get_value(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
        _, _, value = self(obs)  # [batch_size, 1]
        return value  # Keep batch dimension for batched inputs

    def act(self, obs, action_mask, bet_range):
        # obs: [batch_size, obs_dim] or [obs_dim]
        # action_mask: [batch_size, action_dim] or [action_dim]
        # bet_range: [batch_size, 2] or [2] (min_bet, max_bet)
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(self.device)
        bet_range = torch.as_tensor(bet_range, dtype=torch.float32).to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            bet_range = bet_range.unsqueeze(0)

        action_logits, bet_params, value = self(
            obs
        )  # [batch_size, action_dim], [batch_size, 2], [batch_size, 1]

        # Mask invalid actions
        action_logits = action_logits.masked_fill(~action_mask, float("-inf")).clone()
        # none_actions = action_mask.sum(dim=-1) == 0
        # if none_actions.any():
        #     masked_logits[none_actions] = float("-inf")  # overwrite entire row safely
        #     masked_logits[none_actions, Action.PASS.value] = 0.0

        dist = Categorical(logits=action_logits)  # Batch of distributions
        actions = dist.sample()  # [batch_size]
        action_log_probs = dist.log_prob(actions)  # [batch_size]
        entropy = dist.entropy()  # [batch_size]

        # Initialize bet outputs
        batch_size = obs.shape[0]
        bet_sizes = torch.zeros(batch_size, device=self.device)
        bet_log_probs = torch.zeros(batch_size, device=self.device)

        # Handle betting for RAISE actions
        raise_mask = actions == Action.RAISE.value  # [batch_size]
        if raise_mask.any():
            min_bets, max_bets = bet_range[:, 0], bet_range[:, 1]  # [batch_size]
            equal_bets = min_bets == max_bets  # [batch_size]

            # Deterministic case (min_bet == max_bet)
            bet_sizes[equal_bets & raise_mask] = min_bets[equal_bets & raise_mask]
            bet_log_probs[equal_bets & raise_mask] = 0.0

            # Non-deterministic case (min_bet < max_bet)
            non_deterministic = (~equal_bets) & raise_mask
            if non_deterministic.any():
                bet_mean, bet_std = (
                    bet_params[non_deterministic, 0],
                    F.softplus(bet_params[non_deterministic, 1]) + 1e-6,
                )
                base_dist = Normal(bet_mean, bet_std)
                tanh_dist = TransformedDistribution(base_dist, [TanhTransform()])
                raw_samples = tanh_dist.rsample()  # [non_deterministic_sum]
                eps = 1e-6
                safe_samples = raw_samples.clamp(-1 + eps, 1 - eps)
                bet_sizes[non_deterministic] = min_bets[non_deterministic] + 0.5 * (
                    safe_samples + 1
                ) * (max_bets[non_deterministic] - min_bets[non_deterministic])
                bet_log_probs[non_deterministic] = tanh_dist.log_prob(
                    safe_samples
                ) - torch.log(
                    0.5 * (max_bets[non_deterministic] - min_bets[non_deterministic])
                )

        return actions, action_log_probs, bet_sizes, bet_log_probs, entropy, value

    def no_act(self, obs):
        # obs: [batch_size, obs_dim] or [obs_dim]
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        value = self.get_value(obs)  # [batch_size, 1]
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        bet_sizes = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        bet_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        action_log_probs = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        entropy = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        return actions, action_log_probs, bet_sizes, bet_log_probs, entropy, value

    def save(self, path):
        """Save the policy model to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load the policy model from a file."""
        self.load_state_dict(torch.load(path, map_location=self.device))


# --- PPO Buffer ---
class PPOBuffer:
    """Buffer to store experiences for PPO training."""

    def __init__(self, device=torch.device("cpu")):
        self.observations = []
        self.action_masks = []
        self.bet_ranges = []

        self.actions = []
        self.action_log_probs = []
        self.bet_sizes = []
        self.bet_log_probs = []
        self.values = []

        self.rewards = []

        self.advantages = []
        self.returns = []

        self.device = device

    def store(
        self,
        obs,
        action_mask,
        bet_range,
        action,
        action_log_prob,
        bet_size,
        bet_log_prob,
        value,
        reward,
    ):
        self.observations.append(obs)
        self.action_masks.append(action_mask)
        self.bet_ranges.append(bet_range)

        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.bet_sizes.append(bet_size)
        self.bet_log_probs.append(bet_log_prob)
        self.values.append(value)

        self.rewards.append(reward)

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute returns and advantages using GAE (Generalized Advantage Estimation).
        Args:
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.
        Returns:
            None: The buffer is updated in-place with computed returns and advantages.
        """
        rewards = torch.tensor(self.rewards, device=self.device)
        values = torch.stack(self.values).squeeze()
        
        # Vectorized GAE computation
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        
        # Compute advantages using scan operation
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            gae = deltas[t] + gamma * gae_lambda * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        # Normalize advantages per batch, not globally
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()

    def get(self):
        data = {
            "obs": torch.stack(self.observations),
            "actions": torch.stack(self.actions).squeeze(),
            "action_log_probs": torch.stack(self.action_log_probs).squeeze(),
            "bet_sizes": torch.stack(self.bet_sizes).squeeze(),
            "bet_log_probs": torch.stack(self.bet_log_probs).squeeze(),
            "values": torch.stack(self.values).squeeze(),
            "rewards": torch.tensor(self.rewards, device=self.device).squeeze(),
            "returns": torch.tensor(self.returns, device=self.device).squeeze(),
            "advantages": torch.tensor(self.advantages, device=self.device).squeeze(),
            "action_masks": torch.stack(self.action_masks),
            "bet_ranges": torch.stack(self.bet_ranges),
        }
        return data

    def clear(self):
        self.__init__(self.device)  # Reinitialize to clear the buffer


def update_policy(
    policy,
    optimizer,
    data,
    clip_ratio=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    ppo_epochs=4,
):
    policy.train()

    losses = {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0}

    for epoch in range(ppo_epochs):
        optimizer.zero_grad()

        # Extract data
        obs = data["obs"].detach().clone()
        old_actions = data["actions"].detach().clone()
        old_action_log_probs = data["action_log_probs"].detach().clone()
        old_bet_sizes = data["bet_sizes"].detach().clone()
        old_bet_log_probs = data["bet_log_probs"].detach().clone()
        advantages = data["advantages"].detach().clone()
        returns = data["returns"].detach().clone()
        action_masks = data["action_masks"].detach().clone()
        bet_ranges = data["bet_ranges"].detach().clone()

        # Compute new policy outputs
        actions, action_log_probs, bet_sizes, bet_log_probs, entropies, values = (
            policy.act(obs, action_masks, bet_ranges)
        )

        # Compute policy loss
        ratio_input = (action_log_probs - old_action_log_probs) + (
            bet_log_probs - old_bet_log_probs
        )
        ratio_input = torch.clamp(
            ratio_input, min=-20, max=2
        )  # Clamped as we want to avoid extreme values

        ratio = torch.exp(ratio_input)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Compute entropy bonus
        entropy_loss = -entropies.mean()

        # Total loss
        loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        losses["policy_loss"] += policy_loss.item()
        losses["value_loss"] += value_loss.item()
        losses["entropy_loss"] += entropy_loss.item()

    # Average over epochs
    for key in losses:
        losses[key] /= ppo_epochs

    return losses
