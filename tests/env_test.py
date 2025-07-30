"""Unit tests for the PokerGym environment.
This module contains tests for the PokerGym environment, focusing on the
functionality of the environment, including action spaces, observation spaces,
and the handling of different betting rounds.
"""

from pettingzoo.test import api_test

from pokergym.env.poker_env import PokerEnv

def test_api(config, seed=None):
    """
    Test the API of the PokerGym environment.
    This test checks if the environment adheres to the expected API standards.
    """
    env = PokerEnv(config=config,seed=seed)
    env.render_mode = "terminal"
    api_test(env, verbose_progress=True)

if __name__ == "__main__":
    from pokergym.env.config import PokerConfig
    from pokergym.env.enums import BettingRound

    # Create a default configuration for testing
    config = PokerConfig()
    

    # test_api(config, seed=0)  # Run the API test with a seed for reproducibility
    # exit(0)
    import pdb

    try:
        test_api(config,seed=0)  # Uncomment to run the API test
    except Exception as e:
        print(f"An error occurred during the API test: {e}")
        pdb.post_mortem()