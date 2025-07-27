from enum import Enum, auto


class BettingRound(Enum):
    START = -1 # Invalid state, such as before the game starts
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
