"""
Test for the PettingZoo API of the PokerGym environment.
"""

import pytest
from pettingzoo.test import api_test

from pokergym.env.config import PokerConfig
from pokergym.env.texas_holdem import env as PokerEnv

SEEDS = [42, 123, 999, 2025]  # Check different seeds for randomness
CONFIGS = [
    PokerConfig(max_rounds=2),
    PokerConfig(max_rounds=2, num_players=2),
    PokerConfig(max_rounds=2, num_players=3),
    PokerConfig(max_rounds=2, starting_stack=10_000),
    PokerConfig(max_rounds=10),
    PokerConfig(max_rounds=5),
    PokerConfig(max_rounds=5, num_players=2),
    PokerConfig(max_rounds=5, num_players=3),
    PokerConfig(max_rounds=5, starting_stack=10_000),]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_petting_zoo_api(config, seed):
    """
    Uses the built-in PettingZoo API test to ensure the environment adheres to the expected API.
    """
    env = PokerEnv(config=config, seed=seed)
    res = api_test(env, num_cycles=config.num_players * 10_000, verbose_progress=True)
    assert res is None, "API test failed. Please check the environment implementation."
