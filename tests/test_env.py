from pokergym.enviroment.config import PokerConfig
from pokergym.enviroment.enums import BettingRound
from pokergym.enviroment.poker_env import PokerEnv

if __name__ == "__main__":
    config = PokerConfig(num_players=5, starting_stack=1000, big_blind=10, small_blind=5)
    env = PokerEnv(config=config)
    env.render_mode = "human"

    # Simulate a game
    env.reset()

    while env.state.round_number < config.max_rounds:
        print("========================================")
        print(f"Starting round {env.state.round_number + 1}")
        env.start_round()
        # env.render()
        while env.state.betting_round != BettingRound.SHOWDOWN:
            env.step_round()
            # env.render()
        env.render()
        env.end_round()
        