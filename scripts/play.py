
from pokergym.agents import (
    HumanAgent,
    RandomAgent,
    FoldAgent,
)
from pokergym.env.enums import Action, BettingRound
from pokergym.env.texas_holdem import env as PokerEnv
from pokergym.env.utils import action_pretty_str

def player_random_game(config, seed):
    """
    Test a game against with bots that sample moves randomly.
    """
    env = PokerEnv(config=config, seed=seed)
    env.reset(seed=seed)
    agents = [
        RandomAgent(idx=i+1, action_space=env.action_space(i))
        for i in range(config.num_players - 1)
    ]
    # agents.append(HumanAgent(idx=0, action_space=env.action_space(config.num_players - 1), max_chips=env.MAX_CHIPS))
    agents.append(FoldAgent(idx=0, action_space=env.action_space(config.num_players - 1)))
    agents = sorted(agents, key=lambda x: x.idx)
    print(f"Starting game with {len(agents)} agents, you are player {agents[-1].idx}.")
    prev_betting_round = BettingRound.PREFLOP
    for agent in env.agent_iter():
        env.render()
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
        if agent == 0 and action is Action.PASS:
            input("You passed. Press Enter to continue...")
        if env.game_state.betting_round != prev_betting_round:
            prev_betting_round = env.game_state.betting_round
            input(f"New betting round: {prev_betting_round.name}. Press Enter to continue...")


    env.close()

def main():
    from pokergym.env.config import PokerConfig
    config = PokerConfig(num_players=6, starting_stack=1000, small_blind=10, big_blind=20)
    seed = 42
    player_random_game(config, seed)
    print("Game finished.")

if __name__ == "__main__":  
    import pdb
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        pdb.post_mortem()