from pokergym.env.config import PokerConfig
from pokergym.env.enums import BettingRound
from pokergym.env.poker_env import PokerEnv
from pokergym.env.utils import short_pretty_str
from pokergym.agents.base import Agent
from pokergym.env.cards import card_to_int, SeededDeck

import tyro

from deuces import Card



def test_fixed_game(config, starting_cards=None, betting_round_cards=None, seed=None):
    env = PokerEnv(config=config)
    env.render_mode = "terminal"

    # Simulate a game
    env.reset(seed)

    # Start a round with specific cards
    starting_cards = {
        0: [Card.new('Ks'), Card.new('Qd')],
        1: [Card.new('Kh'), Card.new('Qs')],
        2: [Card.new('Kd'), Card.new('Qc')],
        3: [Card.new('Ad'), Card.new('As')],
        4: [Card.new('Ah'), Card.new('Ac')]
    } if starting_cards is None else starting_cards

    env.state.players[3]._chips = 14

    env.start_round(cards=starting_cards)
    # Check if the players received the correct cards
    for i, player in enumerate(env.state.players):
        assert player.hand == starting_cards[i], f"Player {i} did not receive the correct hand."
    env.render()

    print("---Betting Phase---")
    # Force states
    for _ in range(env.state.active_players):
        player = env.state.players[env.state.current_player_idx]
        current_bet = env.state.current_bet
        if player.folded or player.all_in:
            continue
        if player.bet < current_bet:
            bet_amount = current_bet - player.bet
            if player.chips < bet_amount:
                bet_amount = player.chips
            player.make_bet(bet_amount)
        env.render()
        env.state.current_player_idx = env.next_active_player_idx(env.state.current_player_idx)



    print("---Betting Phase Ended---")

    # Give specific cards for the betting rounds
    betting_round_cards = {
        BettingRound.FLOP: [Card.new('2c'), Card.new('3d'), Card.new('5h')],
        BettingRound.TURN: [Card.new('6h')],
        BettingRound.RIVER:[Card.new('8s')]
    } if betting_round_cards is None else betting_round_cards

    # Flop
    env.step_round(cards=betting_round_cards)
    env.state.players[4].folded = True  # Force player 4 to fold for testing

    # Turn
    env.step_round(cards=betting_round_cards)

    # River
    env.step_round(cards=betting_round_cards)

    # Showdown
    env.step_round()


    env.render()
    winner = env.end_round()
    if winner:
        print(f"Game Over! Player {winner.player_id} wins with {winner.chips} chips.")
    env.render()

def test_random_game(config, seed=None):
    env = PokerEnv(config=config)
    env.render_mode = "terminal"

    # Simulate a game
    env.reset(seed)

    while env.state.round_number < config.max_rounds:
        env.start_round()
        # env.render()
        while env.state.betting_round != BettingRound.SHOWDOWN:
            env.step_round()
            # env.render()
        env.render()
        winner = env.end_round()
        if winner:
            print(f"Game Over! Player {winner.idx} wins with {winner.chips} chips.")
            break

def test_agent(config, seed=0):
    env = PokerEnv(config=config)
    env.render_mode = "terminal"
    agents = [Agent(idx=i) for i in range(config.num_players)]
    env.reset(seed)

    obs = env._get_obs(agents[0])  # Initialize observation for the agent
    env.render()
    print(obs)

    print("---Starting Round---")
    env.start_round()
    env.render()
    obs = env._get_obs(agents[0])
    print(obs)

    print("Done")



if __name__ == "__main__":
    # Use tyro to parse command line arguments
    config = tyro.cli(PokerConfig)
    # Example usage:    
    # python -m pokergym.enviroment.poker_env --num_players 5 --starting_stack 1000 --big_blind 100 --small_blind 50 --max_rounds 10000
    # test_fixed_game(config)
    # test_random_game(config, 2)  # Uncomment to run a random game simulation
    test_agent(config, seed=0)  # Uncomment to test an agent
        