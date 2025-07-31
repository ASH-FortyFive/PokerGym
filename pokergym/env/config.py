from dataclasses import dataclass, field
from typing import Optional
from pokergym.env.enums import BettingRound

@dataclass()
class PokerConfig:
    num_players: int = 6
    starting_stack: int = 1000
    big_blind: int = 20
    small_blind: int = 10
    max_community_cards: int = 5
    max_hand_cards: int = 2
    max_rounds: int = 10 
    min_bet: Optional[int] = None # None menas no forced minimum bet
    max_bet: Optional[int] = None # None means no limit
    min_raise: int = 20
    first_dealer: Optional[int] = 0 # None is random

    cards_per_round: dict[BettingRound, int] = field(default_factory=lambda: {
        BettingRound.FLOP: 3,
        BettingRound.TURN: 1,
        BettingRound.RIVER: 1
    })

    # Possibly allow for dynamic betting rounds
    # round_progression: dict[BettingRound, BettingRound] = field(default_factory=lambda: {
    #     BettingRound.START: BettingRound.PREFLOP,
    #     BettingRound.PREFLOP: BettingRound.FLOP,
    #     BettingRound.FLOP: BettingRound.TURN,
    #     BettingRound.TURN: BettingRound.RIVER,
    #     BettingRound.RIVER: BettingRound.SHOWDOWN,
    #     BettingRound.SHOWDOWN: BettingRound.START
    # })