from dataclasses import dataclass, field
from typing import List, Optional

from deuces import Card
from pokergym.env.enums import Action
@dataclass
class Player:
    """A class representing a player in a poker game.
    Attributes:
        idx: Unique identifier for the player.
        _chips: The number of chips the player currently holds (private, use `chips` property).
        hand: List of cards in the player's hand (up to 2 cards for Texas Hold'em).
        folded: Whether the player has folded in the current hand.
        all_in: Whether the player has bet all their chips.
        bet: The player's current bet in the betting round.
        total_contribution: Total chips contributed by the player in the current hand.
        active: Whether the player is still active in the game (e.g., has chips or is all-in).
        last_action: The player's most recent action (e.g., fold, call, raise).
    """
    _idx: int
    _chips: int
    hand: Optional[List[Card]] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    bet: int = 0
    total_contribution: int = 0
    active: bool = True
    last_action: Optional[Action] = None

    def __post_init__(self):
        """Validate initial player state."""
        if self._chips < 0:
            raise ValueError(f"Player {self.idx} cannot have negative chips: {self._chips}")
        if self._chips == 0:
            self.all_in = True

    @property
    def idx(self) -> int:
        """Get the player's unique identifier.
        Returns:
            The player's index.
        """
        return self._idx

    def reset_for_new_hand(self):
        """Reset player state for a new hand.
        Clears the hand, resets folded, all-in, bet, and total contribution status.
        The player's chip count and active status remain unchanged.
        """
        self.hand = []
        self.folded = False
        self.all_in = False
        self.total_contribution = 0
        self.bet = 0
        self.last_action = None

    def add_card(self, card: Card):
        """Add a card to the player's hand.
        Args:
            card: The Card object to add to the player's hand.
        Raises:
            ValueError: If the player's hand already contains 2 cards.
        """
        if len(self.hand) < 2:
            self.hand.append(card)
        else:
            raise ValueError(f"Player {self.idx} cannot add more than 2 cards to hand.")

    @property
    def chips(self) -> int:
        """Get the player's current chip count.
        Returns:
            The number of chips the player has.
        """
        return self._chips

    @chips.setter
    def chips(self, value: int):
        """Set the player's chip count.
        Args:
            value: The new chip count.
        Raises:
            ValueError: If the chip count is negative.
        """
        if value < 0:
            raise ValueError(f"Player {self.idx} cannot have negative chips: {value}")
        self._chips = value
        if self._chips == 0:
            self.all_in = True

    def make_bet(self, amount: int) -> int:
        """Place a bet, deducting chips and updating player state.
        Args:
            amount: The number of chips to bet.
        Returns:
            The player's remaining chip count.
        Raises:
            ValueError: If the bet amount exceeds available chips.
        """
        if amount > self.chips:
            raise ValueError(f"Player {self.idx} cannot bet {amount} chips; only {self.chips} available.")
        self._chips -= amount
        self.total_contribution += amount
        self.bet += amount
        if self._chips == 0:
            self.all_in = True
        return self._chips

    def give_chips(self, amount: int) -> int:
        """Add chips to the player's chip count (e.g., winning a pot).
        Args:
            amount: The number of chips to add.
        Returns:
            The player's new chip count.
        Raises:
            ValueError: If the amount is negative.
        """
        if amount < 0:
            raise ValueError(f"Cannot add negative chips ({amount}) to player {self.idx}.")
        self._chips += amount
        return self._chips