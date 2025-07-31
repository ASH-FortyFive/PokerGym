"""
Test for the PettingZoo API of the PokerGym environment.
"""

import pytest
from pettingzoo.test import api_test

from pokergym.env.config import PokerConfig
from pokergym.env.texas_holdem import env as PokerEnv

SEEDS = [42, 123, 999, 2025]  # Check different seeds for randomness
CONFIGS = [
    PokerConfig(max_hands=2),
    PokerConfig(max_hands=2, num_players=2),
    PokerConfig(max_hands=2, num_players=3),
    PokerConfig(max_hands=2, starting_chips=10_000),
    PokerConfig(max_hands=10),
    PokerConfig(max_hands=5),
    PokerConfig(max_hands=5, num_players=2),
    PokerConfig(max_hands=5, num_players=3),
    PokerConfig(max_hands=5, starting_chips=10_000),]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_petting_zoo_api(config, seed):
    """
    Uses the built-in PettingZoo API test to ensure the environment adheres to the expected API.
    """
    env = PokerEnv(config=config, seed=seed)
    res = api_test(env, num_cycles=config.num_players * 10_000, verbose_progress=True)
    assert res is None, "API test failed. Please check the environment implementation."

if __name__ == "__main__":
    # import sys
    # pytest.main(sys.argv)
    import pdb
    try:
        test_petting_zoo_api(CONFIGS[5], SEEDS[0])  # Run the test directly if this script is executed
    except Exception as e:      
        print(f"An error occurred: {e}")
        pdb.post_mortem()
