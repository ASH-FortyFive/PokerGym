# https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12
# https://huggingface.co/learn/deep-rl-course/unit8/intuition-behind-ppo
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action
from pokergym.env.poker_logic import ActionDict
from pokergym.env.texas_holdem import env as PokerEnv
from pokergym.rl.encoders import ONE_HOT, flatten_space, np_dict_to_torch_dict
from pokergym.rl.ppo.policy import PokerPolicy, PPOBuffer, ppo_update


# --- Env Setup ---
def make_env():
    config = PokerConfig(
        num_players=6, starting_chips=1000, small_blind=10, big_blind=20
    )
    env = PokerEnv(config=config, seed=42, autorender=False)
    env.reset(seed=42)
    return env


# --- Training Loop ---
def train(num_episodes=100, max_steps=6000):
    one_hot = True
    env = make_env()
    obs_space = env.observation_spaces[env.agents[0]]
    action_space = env.action_spaces[env.agents[0]]

    # Initialize policies and optimizers for each agent
    policies = {
        agent: PokerPolicy(
            obs_space, action_space, one_hot=one_hot, device=torch.device("cuda")
        )
        for agent in env.agents
    }
    optimizers = {
        agent: optim.Adam(policies[agent].parameters(), lr=3e-4) for agent in env.agents
    }
    buffers = {agent: PPOBuffer(device="cuda") for agent in env.agents}

    episode_rewards = defaultdict(list)

    ep_pbar = tqdm(total=num_episodes, desc="Training Episodes", unit="episode")
    for ep in range(num_episodes):
        env.reset(seed=42 + ep)  # Vary seed per episode
        done = {agent: False for agent in env.agents}
        ep_rewards = defaultdict(float)
        step = 0

        hand_pbar = tqdm(
            total=max_steps, desc="Steps", unit="step", leave=False, position=1
        )
        for agent in env.agent_iter():
            if all(done.values()) or step >= max_steps:
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

            if termination or truncation:
                action = None
                done[agent] = True
                buffers[agent].store(obs, action, None, None, None, reward)
            elif not action_mask.any():
                # If no valid actions, skip this agent's turn
                action = None
            else:
                action, bet_size, log_prob, entropy, value = policies[agent].act(
                    obs, action_mask, bet_range
                )
                buffers[agent].store(
                    obs=obs,
                    action=action,
                    bet_size=bet_size,
                    log_prob=log_prob,
                    bet_range=bet_range,
                    value=value,
                    reward=reward,
                )
                action = ActionDict(
                    action=action.item(),
                    total_bet=bet_size.item() if bet_size is not None else 0.0,
                )

            hand_over = env.step(action)
            step += 1
            # Update policies
            if info.get("hand_over", False):
                # print(f"Agent {agent} finished hand. Total reward: {ep_rewards[agent]}. Updating policy...")
                # print(f"Active agents: {env.agents}")
                rewards = [r for r in buffers[agent].rewards if r is not None]
                if any(r != 0 for r in rewards):
                    # Compute last value for GAE
                    last_obs = buffers[agent].obs[-1] if buffers[agent].obs else None
                    last_value = (
                        policies[agent]
                        .value_head(policies[agent].backbone(last_obs))
                        .detach()
                        if last_obs is not None
                        else torch.tensor(0.0, device=torch.device("cuda"))
                    )
                    # Compute returns and advantages
                    buffers[agent].compute_returns_and_advantages(
                        last_value=last_value, gamma=0.99, gae_lambda=0.95
                    )
                    # Get data for PPO update
                    data = buffers[agent].get()
                    # Perform PPO update
                    policy_loss, value_loss = ppo_update(
                        policy=policies[agent],
                        optimizer=optimizers[agent],
                        data=data,
                        clip_ratio=0.2,
                        value_loss_coef=0.5,
                        entropy_coef=0.01,
                        epochs=10,
                    )
                    print(
                        f"Agent {agent} updated. Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}"
                    )
                    # Clear buffer for next hand
                    buffers[agent].clear()

            # Update per episode bar
            hand_pbar.update(1)

        # End of episode, update policies
        hand_pbar.close()

        for agent in env.possible_agents:
            episode_rewards[agent].append(ep_rewards[agent])

        ep_pbar.set_postfix(
            episode_rewards={
                agent.split("_")[-1]: float(np.mean(episode_rewards[agent]))
                for agent in env.possible_agents
            }
        )
        ep_pbar.update(1)
    ep_pbar.close()
    return policies, episode_rewards


# --- Run Training ---
if __name__ == "__main__":
    policies, episode_rewards = train()
