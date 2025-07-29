from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import tyro
from deuces import Card

from pokergym.agents import Agent, FishAgent, RandomAgent
from pokergym.env.cards import SeededDeck, card_to_int
from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action, BettingRound
from pokergym.env.poker_env import PokerEnv
from pokergym.env.utils import (
    action_mask_pretty_str,
    action_pretty_str,
    cards_pretty_str,
)


def test_agent(config, seed=0):
    env = PokerEnv(config=config, seed=seed)
    env.render_mode = "terminal"
    agents = [FishAgent(idx=i) for i in range(config.num_players)]
    agents[0] = RandomAgent(
        idx=0, reasonable_raises=True
    )  # Use a random agent for player 0
    agents[5] = RandomAgent(
        idx=5, reasonable_raises=True
    )  # Use a random agent for player 1
    np.random.seed(seed=seed)

    env.reset(seed)
    env.render()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(env.terminations)
        print(env.agents)
        action_mask = observation["action_mask"]
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask=action_mask)
        # print(f"Player {agent}, Round {env.state.round_number} ({env.state.betting_round.name}), action: {action_pretty_str(action, max_chips=env.MAX_CHIPS)}, action mask: {action_mask_pretty_str(action_mask, max_chips=env.MAX_CHIPS)}")
        # print(
            # f"Player {agent}, action: {action_pretty_str(action, max_chips=env.MAX_CHIPS)}"
        # )
        env.step(action)
        env.render()
    env.close()

    print("Game Over!")

if __name__ == "__main__":
    # Define a wrapper to include optional seed
    @dataclass
    class ExtraArgs:
        config: PokerConfig
        seed: Optional[int] = 0  # Default seed value

    # Use tyro to parse command line arguments
    args = tyro.cli(ExtraArgs)
    args.config.max_rounds = 1

    # Example usage:
    # python -m pokergym.enviroment.poker_env --num_players 5 --starting_stack 1000 --big_blind 100 --small_blind 50 --max_rounds 10000
    # test_fixed_game(config)
    # test_random_game(config, 2)  # Uncomment to run a random game simulation
    test_agent(args.config, seed=args.seed)  # Uncomment to test an agent
    # exit(0)
    # import pdb
    # try:
    #     test_agent(args.config, seed=args.seed)  # Uncomment to test an agent
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     pdb.post_mortem()
