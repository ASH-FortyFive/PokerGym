# https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12
# https://huggingface.co/learn/deep-rl-course/unit8/intuition-behind-ppo
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from pokergym.agents.random import RandomAgent
from pokergym.env.config import PokerConfig
from pokergym.env.poker_logic import ActionDict
from pokergym.env.texas_holdem import env as PokerEnv
from pokergym.rl.encoders import ONE_HOT, flatten_space, np_dict_to_torch_dict
from pokergym.rl.ppo.policy import PokerPolicy, PPOBuffer, update_policy
from pokergym.rl.utils import set_seed

have_wandb = False
try:
    import wandb

    have_wandb = True
except ImportError:
    print("Weights & Biases not installed. Skipping W&B logging.")


@dataclass()
class TrainingConfig:
    """Configuration for the Training process."""

    num_episodes: int = 10_000
    max_steps: int = 6000
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 42
    ppo_epochs: int = 4


@dataclass()
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    project: str = "PokerGym"
    entity: str = field(
        default_factory=lambda: os.getenv("WANDB_ENTITY", "as03095-surrey")
    )
    group: str = "ppo_training"
    name: str = "ppo_poker_training"
    use: bool = True  # Set to False to disable W&B logging


@dataclass()
class Config:
    """Configuration for the training script."""

    poker: PokerConfig = field(default_factory=PokerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    one_hot: bool = True
    save: bool = True  # Whether to save the model while training
    save_every_n_episodes: int = 10
    device: str = "cuda"
    save_path: str = "checkpoints"  # Path to save the trained policy
    test: bool = True  # Whether to run a tests against random agents during training


# --- Env Setup ---
def make_env(poker_config: PokerConfig, seed: Optional[int] = None):
    env = PokerEnv(config=poker_config, seed=seed, autorender=False)
    env.reset(seed=seed)
    return env


# --- Training Loop ---
def train(config: Config):
    one_hot = config.one_hot
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    env = make_env(config.poker, seed=config.training.seed)
    obs_space = env.observation_spaces[env.agents[0]]
    action_space = env.action_spaces[env.agents[0]]

    # Initialize policies and optimizers for each agent
    policies = {
        agent: PokerPolicy(obs_space, action_space, one_hot=one_hot, device=device)
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

    ep_pbar = tqdm(
        total=config.training.num_episodes,
        desc="Training Episodes",
        unit="episode",
    )
    for ep in range(config.training.num_episodes):
        env.reset(seed=42 * ep)  # Vary seed per episode
        done = {agent: False for agent in env.agents}
        ep_rewards = defaultdict(float)
        step = 0

        tournament_pbar = tqdm(
            total=config.training.max_steps,
            desc="Train steps",
            unit="step",
            leave=False,
            position=1,
        )
        for agent in env.agent_iter():
            if all(done.values()) or step >= config.training.max_steps:
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

            if torch.any(torch.isnan(action_mask)):
                print(f"NaN detected in action mask for agent {agent}. Skipping step.")
                continue

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
                gamma=config.training.gamma,
                gae_lambda=config.training.gae_lambda,
            )
            data = buffers[agent].get()
            if torch.isnan(data["action_masks"]).any():
                print(f"NaN detected in actions for agent {agent}. Skipping update.")
                continue

            losses = update_policy(
                policy=policies[agent],
                optimizer=optimizers[agent],
                data=data,
                clip_ratio=config.training.clip_ratio,  # PPO clipping parameter
                value_loss_coef=config.training.value_loss_coef,
                entropy_coef=config.training.entropy_coef,
                max_grad_norm=0.5,  # Gradient clipping
                ppo_epochs=config.training.ppo_epochs,
            )

            if have_wandb and config.wandb.use:
                wandb.log(
                    {
                        f"loss/policy_loss/{agent}": losses["policy_loss"],
                        f"loss/value_loss/{agent}": losses["value_loss"],
                        f"loss/entropy/{agent}": losses["entropy_loss"],
                        f"reward/{agent}": np.mean(episode_rewards[agent]),
                    },
                    step=ep,
                )

            buffers[agent].clear()  # Clear buffer after update

        # Every X episodes, play against agents that sample moves randomly, to report performance
        if (ep + 1) % 2 == 0 and config.test:
            chips_v_random = 0.0
            num_trials = 10
            test_pbar = tqdm(
                total=num_trials,
                desc="Test Episodes",
                unit="trial",
                leave=False,
                position=1,
            )
            for trial in range(num_trials):
                env.reset(seed=42 + trial + ep)  # Reset with a new seed
                step = 0
                done = {agent: False for agent in env.agents}
                for agent in env.agent_iter():
                    if all(done.values()) or step >= config.training.max_steps:
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
                                total_bet=(
                                    bet_size.item() if bet_size is not None else 0.0
                                ),
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
                chips_v_random += int(
                    env.poker.game_state.players[test_player_idx].chips
                )
            test_pbar.close()
            chips_v_random /= num_trials

        if (ep + 1) % 10 == 0 or ep == config.training.num_episodes - 1 and config.save:
            # Save model every 10 episodes
            for agent, policy in policies.items():
                path = f"{config.save_path}/policy_{agent}.pth"
                policy.save(path)

        # Update progress bar
        avg_rewards = {
            f"reward/{agent}": np.mean(episode_rewards[agent][-1:])
            for agent in env.possible_agents
        }
        if have_wandb and config.wandb.use:
            wandb.log({**avg_rewards, "Chips_vs_Random": chips_v_random, "episode": ep})
        ep_pbar.set_postfix({"Chips vs Random": chips_v_random})

        ep_pbar.update(1)
    ep_pbar.close()
    return policies, episode_rewards


# --- Run Training ---
if __name__ == "__main__":
    import tyro

    config = tyro.cli(Config)

    os.makedirs(config.save_path, exist_ok=True)
    # config.wandb.use = False
    # config.test = False  # Disable testing by default
    # config.save = False
    if config.wandb.use and have_wandb:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            config=config,
        )

    set_seed(config.training.seed)
    policies, episode_rewards = train(config)

    if config.wandb.use and have_wandb:
        wandb.finish()
