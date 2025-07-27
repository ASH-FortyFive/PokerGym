from dataclasses import dataclass, field
from typing import List, Optional

from deuces import Card


@dataclass
class Player:
    player_id: int
    chips: int
    hand: Optional[List[Card]] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    bet: int = 0

    def reset_for_new_hand(self):
        self.hand = []
        self.folded = False
        self.all_in = False
        self.bet = 0
