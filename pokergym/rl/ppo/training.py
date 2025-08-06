# https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12
# https://huggingface.co/learn/deep-rl-course/unit8/intuition-behind-ppo
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
# import wandb
from tqdm import tqdm

from pokergym.agents.random import RandomAgent
from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action
from pokergym.env.poker_logic import ActionDict
from pokergym.env.texas_holdem import env as PokerEnv
from pokergym.rl.encoders import ONE_HOT, flatten_space, np_dict_to_torch_dict
from pokergym.rl.ppo.policy import PokerPolicy, PPOBuffer, update_policy
from dataclasses import dataclass, field
from typing import Optional
from pokergym.env.enums import BettingRound


@dataclass()
class TrainingConfig:
    """Configuration for the Training process."""
    num_episodes: int = 100
    max_steps: int = 6000
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 42

@dataclass()
class Config:
    """Configuration for the training script."""
    poker_config: PokerConfig = field(default_factory=PokerConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    one_hot: bool = True
    save_every_n_episodes: int = 10
    device: str = "cuda"
    save_path: str = "checkpoints"  # Path to save the trained policy

# --- Env Setup ---
def make_env(poker_config: PokerConfig, seed: Optional[int] = None):
    env = PokerEnv(config=poker_config, seed=seed, autorender=False)
    env.reset(seed=seed)
    return env


# --- Training Loop ---
def train(config: Config):
    one_hot = config.one_hot
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    env = make_env(config.poker_config, seed=config.training_config.seed)
    obs_space = env.observation_spaces[env.agents[0]]
    action_space = env.action_spaces[env.agents[0]]

    # Initialize policies and optimizers for each agent
    policies = {
        agent: PokerPolicy(
            obs_space, action_space, one_hot=one_hot, device=device
        )
        for agent in env.agents
    }
    optimizers = {
        agent: optim.Adam(policies[agent].parameters(), lr=3e-4) for agent in env.agents
    }
    buffers = {agent: PPOBuffer(device="cuda") for agent in env.agents}

    # Initialize random agent
    random_agent = RandomAgent(idx=-1, action_space=env.action_spaces[env.agents[-1]])
    chips_v_random = 0.0  # Placeholder for random agent's value

    episode_rewards = defaultdict(list)

    ep_pbar = tqdm(total=config.training_config.num_episodes, desc="Training Episodes", unit="episode")
    for ep in range(config.training_config.num_episodes):
        env.reset(seed=42 * ep)  # Vary seed per episode
        done = {agent: False for agent in env.agents}
        ep_rewards = defaultdict(float)
        step = 0

        tournament_pbar = tqdm(
            total=config.training_config.max_steps, desc="Train steps", unit="step", leave=False, position=1
        )
        for agent in env.agent_iter():
            if all(done.values()) or step >= config.training_config.max_steps:
                break

            observation, reward, termination, truncation, info = env.last()
            ep_rewards[agent] += reward

            obs = np_dict_to_torch_dict(
                observation,
                one_hot=ONE_HOT if one_hot else {},
                device=torch.device("cuda"),
            )
            action_mask = obs["action_mask"]["action"]
            bet_range = obs["action_mask"]["total_bet"]
            obs = flatten_space(obs, device=torch.device("cuda"))
            passing = True

            if termination or truncation:
                done[agent] = True

            if (termination or truncation) or not action_mask.any():
                env_action = None
                action, action_log_prob, bet_size, bet_log_prob, entropy, value = (
                    policies[agent].no_act(obs)
                )
                action_mask[-1] = 1 
            else:
                action, action_log_prob, bet_size, bet_log_prob, entropy, value = (
                    policies[agent].act(obs, action_mask, bet_range)
                )
                env_action = ActionDict(
                    action=action.item(),
                    total_bet=bet_size.item() if bet_size is not None else 0.0,
                )
                passing = False

            # Log action and value
            step += 1
            env.step(env_action)

            # Store experience in buffer
            if not passing or reward != 0:
                buffers[agent].store(
                    obs=obs,
                    action_mask=action_mask,
                    bet_range=bet_range,
                    action=action,
                    action_log_prob=action_log_prob,
                    bet_size=bet_size,
                    bet_log_prob=bet_log_prob,
                    value=value,
                    reward=reward,
                )

            # Update per episode bar
            tournament_pbar.update(1)
        # End of episode, update policies
        tournament_pbar.close()

        for agent in env.possible_agents:
            episode_rewards[agent].append(ep_rewards[agent])
            # Update policy for the agent
            # print(f"Hand over for agent {agent}, updating policy.")
            if not buffers[agent].rewards:
                print(f"No rewards collected for agent {agent}, skipping update.")
                continue

            buffers[agent].compute_returns_and_advantages(
                gamma=config.training_config.gamma, gae_lambda=config.training_config.gae_lambda, last_value=0.0
            )
            update_policy(
                policy=policies[agent],
                optimizer=optimizers[agent],
                data=buffers[agent].get(),
                clip_ratio=config.training_config.clip_ratio,  # PPO clipping parameter
                value_loss_coef=config.training_config.value_loss_coef,
                entropy_coef=config.training_config.entropy_coef,
            )
            buffers[agent].clear()  # Clear buffer after update

        # Every X episodes, play against agents that sample moves randomly, to report performance
        if (ep + 1) % 2 == 0:
            chips_v_random = 0.0
            num_trials = 10
            test_pbar = tqdm(
                total=num_trials, desc="Test Episodes", unit="trial", leave=False, position=1
            )
            for trial in range(num_trials):
                env.reset(seed=42 + trial + ep)  # Reset with a new seed
                step = 0
                done = {agent: False for agent in env.agents}
                for agent in env.agent_iter():
                    if all(done.values()) or step >= config.training_config.max_steps:
                        break

                    observation, reward, termination, truncation, info = env.last()

                    obs = np_dict_to_torch_dict(
                        observation,
                        one_hot=ONE_HOT if one_hot else {},
                        device=torch.device("cuda"),
                    )
                    action_mask = obs["action_mask"]["action"]
                    bet_range = obs["action_mask"]["total_bet"]
                    obs = flatten_space(obs, device=torch.device("cuda"))

                    if termination or truncation:
                        done[agent] = True

                    if (termination or truncation) or not action_mask.any():
                        env_action = None
                    else:
                        if agent == env.possible_agents[0]:
                            (
                                action,
                                action_log_prob,
                                bet_size,
                                bet_log_prob,
                                entropy,
                                value,
                            ) = policies[agent].act(obs, action_mask, bet_range)
                            env_action = ActionDict(
                                action=action.item(),
                                total_bet=bet_size.item() if bet_size is not None else 0.0,
                            )
                        else:
                            bot_action_mask = {
                                "action": action_mask.detach().cpu().clone().numpy(),
                                "total_bet": bet_range.detach().cpu().clone().numpy(),
                            }
                            env_action = random_agent.act(observation, bot_action_mask)
                    # Log action and value
                    step += 1
                    env.step(env_action)
                test_pbar.update(1)
                test_player_idx = env.agent_name_mapping[env.possible_agents[0]]
                chips_v_random += int(env.poker.game_state.players[test_player_idx].chips)
            test_pbar.close()
            chips_v_random /= num_trials

        if (ep + 1) % 10 == 0:
            # Save model every 10 episodes
            for agent, policy in policies.items():
                path = f"{config.save_path}/policy_{agent}.pth"
                policy.save(path)

        # Update progress bar
        ep_pbar.set_postfix({"Chips vs Random": chips_v_random})

        ep_pbar.update(1)
    ep_pbar.close()
    return policies, episode_rewards


# --- Run Training ---
if __name__ == "__main__":
    import tyro 
    config = tyro.cli(Config)

    policies, episode_rewards = train(config)
