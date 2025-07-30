from enum import Enum, auto


class BettingRound(Enum):
    START = 0
    PREFLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    SHOWDOWN = 5
    END = 6

class Action(Enum):
    FOLD = 0
    CALL = 1
    CHECK = 2
    RAISE = 3
    PASS = 4 # Used to skip the turn without betting, only available if inactive, folded, or all-in
