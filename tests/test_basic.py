import pytest
from pokergym.env.config import PokerConfig
from pokergym.env.poker_env import PokerEnv


def test_env_creation():
    """Test that the environment can be created."""
    config = PokerConfig(num_players=2)
    env = PokerEnv(config=config)
    assert env is not None
    assert len(env.possible_agents) == 2


def test_env_reset():
    """Test that the environment can be reset."""
    config = PokerConfig(num_players=2)
    env = PokerEnv(config=config)
    env.reset()
    assert env.agent_selection is not None


def test_config_creation():
    """Test that config can be created with default values."""
    config = PokerConfig()
    assert config.num_players == 5
    assert config.starting_stack == 100


if __name__ == "__main__":
    test_env_creation()
    test_env_reset()
    test_config_creation()
    print("All tests passed!")