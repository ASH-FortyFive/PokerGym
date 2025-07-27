from dataclasses import dataclass, field
from typing import List, Optional

from deuces import Card


@dataclass
class Player:
    player_id: int
    _chips: int
    hand: Optional[List[Card]] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    total_contribution: int = 0

    def reset_for_new_hand(self):
        self.hand = []
        self.folded = False
        self.all_in = False
        self.total_contribution = 0

    def add_card(self, card: Card):
        if len(self.hand) < 2:
            self.hand.append(card)
        else:
            raise ValueError("Cannot add more than 2 cards to a player's hand.")
        
    @property
    def chips(self) -> int:
        return self._chips  
    
    @chips.setter
    def chips(self, value: int):
        if value < 0:
            raise ValueError("Chips cannot be negative.")
        self._chips = value
        if self._chips == 0:
            self.all_in = True
    
    def make_bet(self, amount: int):
        if amount > self.chips:
            raise ValueError("Cannot bet more than available chips.")
        self._chips -= amount
        self.total_contribution += amount
        if self._chips == 0:
            self.all_in = True
        return self._chips

    def give_chips(self, amount: int):
        if amount < 0:
            raise ValueError("Cannot give negative chips.")
        self._chips += amount
        return self._chips 
