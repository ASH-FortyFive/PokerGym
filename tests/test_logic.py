import pytest
from pokergym.env.poker_logic import PokerLogic, PokerGameState

STATE_1 = PokerGameState(1_000, 6, 0, 42)
STATES = [STATE_1]
EXPECTIONS =[None]

@pytest.mark.parametrize("game_state", STATES)
@pytest.mark.parametrize("expected", EXPECTIONS)
def test_legal_actions(game_state, expected):
    """
    Test the legal_actions method of PokerLogic.
    """
    
