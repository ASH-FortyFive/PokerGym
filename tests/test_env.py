import numpy as np
import pytest
from deuces import Card

from pokergym.agents import (
    Agent,
    CallAgent,
    CheckAgent,
    FoldAgent,
    RaiseAgent,
    RandomAgent,
)
from pokergym.env.cards import SeededDeck, card_to_int
from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action, BettingRound
from pokergym.env.texas_holdem import env as PokerEnv
from pokergym.env.utils import (
    action_mask_pretty_str,
    action_pretty_str,
    cards_pretty_str,
)

SEEDS = [42, 0, 100]
CONFIGS = [
    PokerConfig(max_rounds=2),
    PokerConfig(max_rounds=2, num_players=2,first_dealer=1),
    PokerConfig(max_rounds=2, num_players=3),
    PokerConfig(max_rounds=2, starting_stack=10_000),
    PokerConfig(max_rounds=10),
    PokerConfig(max_rounds=5, first_dealer=None),
    PokerConfig(max_rounds=5, num_players=2, first_dealer=0),
    PokerConfig(max_rounds=5, num_players=3),
    PokerConfig(max_rounds=5, starting_stack=10_000),]

# Format for fixing cards
OPTIONS = [
    {
        "cards": {
            0: {
                BettingRound.START: {
                    0: [Card.new("As"), Card.new("Ad")],
                    1: [Card.new("Ac"), Card.new("Ah")],
                    2: [Card.new("2s"), Card.new("2d")],
                    3: [Card.new("3c"), Card.new("3h")],
                    4: [Card.new("4s"), Card.new("4d")],
                    5: [Card.new("5c"), Card.new("5h")],
                },
                BettingRound.FLOP: [
                    Card.new("2h"),
                    Card.new("6s"),
                    Card.new("7d"),
                    Card.new("8c"),
                ],  # First card is the burn card
                BettingRound.TURN: [Card.new("7h"), Card.new("9h")],
                BettingRound.RIVER: [Card.new("8h"), Card.new("Th")],
            },
            1: {
                BettingRound.START: {
                    0: [Card.new("9s"), Card.new("9d")],
                    1: [Card.new("8c"), Card.new("8h")],
                    2: [Card.new("7s"), Card.new("7d")],
                    3: [Card.new("6c"), Card.new("6h")],
                    4: [Card.new("5s"), Card.new("5d")],
                    5: [Card.new("4c"), Card.new("4h")],
                },
                BettingRound.FLOP: [
                    Card.new("2h"),
                    Card.new("3s"),
                    Card.new("2d"),
                    Card.new("As"),
                ],
                BettingRound.TURN: [Card.new("7h"), Card.new("Ad")],
                BettingRound.RIVER: [Card.new("9h"), Card.new("Ac")],
            },
        }
    }
]
@pytest.mark.parametrize("config", CONFIGS[:4])
@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("outcome", OUTCOMES) TODO: Implement outcome tests
def test_fixed_game(config: PokerConfig, options, seed): #, outcome):
    """
    Test a fixed game with predetermined cards.
    """
    env = PokerEnv(config=config, seed=seed)
    env.reset(seed=seed, options=options)
    env.render_mode = "terminal"
    prev_round = -1
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action_mask = observation["action_mask"]
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample(mask=action_mask)
        env.step(action)
        if env.game_state.round_number != prev_round:
            prev_round = env.game_state.round_number
            sanity_checks(env, config)
    sanity_checks(env, config)
    env.close()


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_fold_game(config: PokerConfig, seed):
    """
    Test a game with bots that only fold.
    """
    env = PokerEnv(config=config, seed=seed)
    env.reset(seed=seed)
    env.render_mode = "terminal"
    agents = [
        FoldAgent(idx=i, action_space=env.action_space(i))
        for i in range(config.num_players)
    ]
    prev_round = -1
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action_mask = observation["action_mask"]
        if termination or truncation:
            action = None
        else:
            action = agents[agent].act(observation, action_mask)
        env.step(action)
        if env.game_state.round_number != prev_round:
            prev_round = env.game_state.round_number
            sanity_checks(env, config)
    sanity_checks(env, config)
    env.close()

@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_random_game(config, seed):
    """
    Test a game with bots that sample moves randomly.
    """
    env = PokerEnv(config=config, seed=seed)
    env.reset(seed=seed)
    env.render_mode = "terminal"
    assert env.game_state.dealer_idx == config.first_dealer if config.first_dealer is not None else True, f"Dealer index should be set correctly. Wanted {config.first_dealer}, got {env.game_state.dealer_idx}"
    assert env.game_state.round_number == 0, f"Round number should start at 0. Got {env.game_state.round_number}"
    assert len(env.game_state.players) == config.num_players, f"Number of players should be {config.num_players}. Got {len(env.game_state.players)}"
    sanity_checks(env, config)
    agents = [
        RandomAgent(idx=i, action_space=env.action_space(i))
        for i in range(config.num_players)
    ]
    prev_round = -1
    env.render()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action_mask = observation["action_mask"]
        if termination or truncation:
            action = None
        else:
            action = agents[agent].act(observation, action_mask)
        env.step(action)
        print(
            f"Player {agent}, action: {action_pretty_str(action, max_chips=env.MAX_CHIPS)}"
        )
        env.render()


        if env.game_state.round_number != prev_round:
            prev_round = env.game_state.round_number
            sanity_checks(env, config)
    sanity_checks(env, config)
    env.close()


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rules_game(config, seed):
    """
    Test a game with bots that sample moves randomly.
    """
    np.random.seed(seed)
    env = PokerEnv(config=config, seed=seed)
    env.reset(seed=seed)
    env.render_mode = "terminal"
    assert env.game_state.dealer_idx == config.first_dealer if config.first_dealer is not None else True, f"Dealer index should be set correctly. Wanted {config.first_dealer}, got {env.game_state.dealer_idx}"
    assert env.game_state.round_number == 0, f"Round number should start at 0. Got {env.game_state.round_number}"
    assert len(env.game_state.players) == config.num_players, f"Number of players should be {config.num_players}. Got {len(env.game_state.players)}"
    sanity_checks(env, config)
    agent_options = [RaiseAgent, CallAgent, CheckAgent, FoldAgent, RandomAgent]
    # Sample randomly from the agent options
    agents = [
        agent_options[np.random.randint(len(agent_options))](
            idx=i, action_space=env.action_space(i)
        ) for i in range(config.num_players)
    ]
    for agent in agents:
        if isinstance(agent, RandomAgent):
            agent.reasonable_raises = np.random.choice([True, False])

    prev_round = -1
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action_mask = observation["action_mask"]
        if termination or truncation:
            action = None
        else:
            action = agents[agent].act(observation, action_mask)
        env.step(action)
        if env.game_state.round_number != prev_round:
            prev_round = env.game_state.round_number
            sanity_checks(env, config)
    sanity_checks(env, config)
    env.close()

def sanity_checks(env: PokerEnv, config: PokerConfig):
    """Ensure the environment is in a valid state functioning correctly."""
    total_chips = config.num_players * config.starting_stack
    total_chips_in_state = 0
    total_chips_in_state += sum(player.chips for player in env.game_state.players)
    total_chips_in_state += sum(player.bet for player in env.game_state.players)
    total_chips_in_state += env.game_state.pot
    assert (
        total_chips == total_chips_in_state
    ), f"Total chips mismatch: expected {total_chips}, got {total_chips_in_state}"
    assert (
        len(env.game_state.players) == config.num_players
    ), f"Number of players mismatch: expected {config.num_players}, got {len(env.game_state.players)}"

if __name__ == "__main__":
    # import sys
    # pytest.main(sys.argv)
    import pdb
    try:
        test_random_game(CONFIGS[4], SEEDS[2])  # Run the test directly if this script is executed
    except Exception as e:      
        print(f"An error occurred: {e}")
        pdb.post_mortem()