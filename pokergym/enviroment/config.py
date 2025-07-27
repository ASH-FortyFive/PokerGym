from dataclasses import dataclass


@dataclass(frozen=True)
class PokerConfig:
    num_players: int = 2
    starting_stack: int = 100
    big_blind: int = 2
    small_blind: int = 1
    max_community_cards: int = 5
    max_hand_cards: int = 2
    max_rounds: int = 10 
    max_bet: int = None # None means no limit
