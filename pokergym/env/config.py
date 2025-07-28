from dataclasses import dataclass, field
from pokergym.env.enums import BettingRound

@dataclass(frozen=True)
class PokerConfig:
    num_players: int = 5
    starting_stack: int = 1000
    big_blind: int = 20
    small_blind: int = 10
    max_community_cards: int = 5
    max_hand_cards: int = 2
    max_rounds: int = 10 
    min_bet: int = None # None menas no forced minimum bet
    max_bet: int = None # None means no limit
    min_raise: int = 10

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