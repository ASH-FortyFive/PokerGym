from enum import Enum, auto


class BettingRound(Enum):
    START = auto() # Invalid state, such as before the game starts
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()
    END = auto()  # Represents the end of the game or round

class Action(Enum):
    FOLD = 0
    CALL = 1
    CHECK = 2
    RAISE = 3
    PASS = 4 # Used to skip the turn without betting, only available if inactive, folded, or all-in
