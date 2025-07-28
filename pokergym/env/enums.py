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
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
