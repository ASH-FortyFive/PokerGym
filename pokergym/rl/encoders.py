from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Discrete, MultiDiscrete

from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action, BettingRound
from pokergym.env.texas_holdem import env as PokerEnv

ONE_HOT = {
    "community_cards": 53,
    "hand": 53,
    "betting_round": max([br.value for br in BettingRound]) + 1,
    # "action_mask": max([action.value for action in Action]) + 1,
    "action": max([action.value for action in Action]) + 1,
}


def np_dict_to_torch_dict(
    obs: Dict, one_hot: dict = {}, device: torch.device = "cpu"
) -> torch.Tensor:
    """Convert dictionary of numpy arrays to a dictionary of PyTorch tensors.
    Args:
        obs (dict): Dictionary with numpy arrays.
        one_hot (dict): Dictionary of keys to convert to one-hot encoding with their number of classes.
        device (torch.device): Device to place the tensors on.
    """
    obs_tensor = {}
    for key, val in obs.items():
        # print(f"Processing key: {key}, value type: {type(value)} {value.dtype if hasattr(value, 'dtype') else ''}")
        if isinstance(val, dict):
            obs_tensor[key] = np_dict_to_torch_dict(val, device=device)
        elif key in one_hot:
            n_classes = one_hot[key]
            size = val.size if isinstance(val, np.ndarray) else 1
            one_hot_tensor = torch.zeros((size, n_classes), device=device)
            if isinstance(val, np.ndarray):
                # Convert to one-hot encoding
                one_hot_tensor[torch.arange(val.size), val] = 1.0
                obs_tensor[key] = one_hot_tensor
            elif isinstance(val, np.int64) or isinstance(val, np.int32) or isinstance(val, int):
                # Single integer value, convert to one-hot
                one_hot_tensor[0, val] = 1.0
                obs_tensor[key] = one_hot_tensor
            else:
                raise ValueError(
                    f"Expected numpy array for one-hot encoding, got {type(val)} for key {key}"
                )
        else:
            obs_tensor[key] = torch.tensor(val, device=device)
    return obs_tensor

def flatten_space(obs, device="cpu"):
    """Flatten and concatenate observation dict."""
    features = []
    if isinstance(obs, torch.Tensor):
        return obs.flatten().to(device)
    for key, val in obs.items():
        if isinstance(val, dict):
            new_feature = flatten_space(val)
        else:
            new_feature = val.flatten()
        features.append(new_feature)
    device = features[0].device if features else device
    return torch.cat(features, dim=-1).float().to(device)



class BackboneNetwork(nn.Module):
    def __init__(
        self,
        observation_space,
        hidden_dimensions,
        out_features,
        dropout=0.1,
        one_hot: bool = True,
        device="cuda",
    ):
        """
        Backbone network for processing observation space in PokerGym.
        Args:
            observation_space (dict): Dictionary representing the observation space.
            hidden_dimensions (int): Number of hidden dimensions for the network.
            out_features (int): Number of output features.
            dropout (float): Dropout rate for regularization.
            device (str): Device to run the model on, default is "cuda".
        """
        super().__init__()
        self.one_hot = one_hot
        self.in_features = self._compute_input_dim(observation_space)
        self.layer1 = nn.Linear(self.in_features, hidden_dimensions, device=device)
        self.layer2 = nn.Linear(hidden_dimensions, out_features, device=device)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.to(device)

    def _compute_input_dim(self, observation_space):
        """Compute total flattened feature length."""
        dims = 0
        for key, val in observation_space.items():
            if isinstance(val, dict) or isinstance(val, GymDict):
                new_dim = self._compute_input_dim(val)
            # For discrete spaces, make one-hot encoding
            elif self.one_hot and isinstance(val, MultiDiscrete):
                new_dim = np.sum(val.nvec)
            elif self.one_hot and isinstance(val, Discrete):
                shape = val.shape[0] if len(val.shape) > 0 else 1
                new_dim = val.n * shape
            else:
                new_dim = val.shape[0] if len(val.shape) > 0 else 1
            dims += new_dim
        return dims

    def forward(self, obs):
        # one_hot = ONE_HOT if self.one_hot else None
        # obs_torch = np_dict_to_torch_dict(obs, one_hot=one_hot, device=self.device)
        x = self.layer1(obs)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        return x


def make_env():
    config = PokerConfig(
        num_players=6, starting_chips=1000, small_blind=10, big_blind=20
    )
    env = PokerEnv(config=config, seed=42, autorender=False)
    env.reset(seed=42)
    return env

if __name__ == "__main__":
    one_hot = True

    env = make_env()
    obs_space = env.observation_spaces[env.agents[0]]
    obs = obs_space.sample()
    obs_tensor = np_dict_to_torch_dict(
        obs, one_hot=ONE_HOT if one_hot else {}, device=torch.device("cuda")
    )
    obs_flat = flatten_space(obs_tensor, device=torch.device("cuda"))
    action_space = env.action_spaces[env.agents[0]]
    action_mask = obs["action_mask"]
    if action_mask["total_bet"][0] > action_mask["total_bet"][1]:
        # Flip
        temp = action_mask["total_bet"][0]
        action_mask["total_bet"][0] = action_mask["total_bet"][1]
        action_mask["total_bet"][1] = temp
    action = action_space.sample(mask=action_mask)
    action_tensor = np_dict_to_torch_dict(action, device=torch.device("cuda"))
    # Initialize the backbone network
    backbone = BackboneNetwork(
        obs_space, hidden_dimensions=128, out_features=64, dropout=0.1
    )
    backbone = backbone(obs_flat)