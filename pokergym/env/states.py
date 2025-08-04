from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deuces import Card
from pokergym.env.cards import SeededDeck, card_to_str
from pokergym.env.enums import Action, BettingRound
import numpy as np


@dataclass
class PlayerState:
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
        """Initialize defaults and validate chips."""
        self.hand = self.hand or []
        if self.chips < 0:
            raise ValueError(f"Player {self.idx} cannot have negative chips: {self.chips}")
        if self.chips == 0:
            self.all_in = True

    @property
    def idx(self) -> int:
        """Get the player's unique identifier."""
        return self._idx

    def reset_for_new_hand(self):
        """Reset player state for a new hand."""
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
        """
        if len(self.hand) < 2:
            self.hand.append(card)
        else:
            raise ValueError(f"Player {self.idx} cannot add more than 2 cards to hand.")

    @property
    def chips(self) -> int:
        """Get the player's current chip count."""
        return self._chips

    @chips.setter
    def chips(self, value: int):
        """Set the player's chip count.
        Args:
            value: The new chip count.
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
        """
        if amount < 0:
            raise ValueError(f"Cannot add negative chips ({amount}) to player {self.idx}.")
        self._chips += amount
    
    def to_json(self) -> Dict[str, Any]:
        """Convert player state to a JSON-compatible dictionary."""
        return {
            "idx": self.idx,
            "chips": self.chips,
            "hand": [card_to_str(card) for card in self.hand],  # Convert Card to string
            "folded": self.folded,
            "all_in": self.all_in,
            "bet": self.bet,
            "total_contribution": self.total_contribution,
            "active": self.active,
            "last_action": self.last_action.name if self.last_action else None  # Convert Action enum to name
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PlayerState':
        """Reconstruct player state from a JSON dictionary."""
        hand = [Card.new(card_str) for card_str in data.get("hand", [])]  # Convert strings back to Card
        last_action = Action[data["last_action"]] if data.get("last_action") else None  # Convert name to Action
        return cls(
            _idx=data["idx"],
            _chips=data["chips"],
            hand=hand,
            folded=data["folded"],
            all_in=data["all_in"],
            bet=data["bet"],
            total_contribution=data["total_contribution"],
            active=data["active"],
            last_action=last_action
        )

@dataclass
class PokerGameState:
    """Dataclass to hold the state of a poker game.
    This encapsulates all essential game state information, including player details,
    betting status, community cards, and deck state, providing a comprehensive snapshot
    of the game at any point.
    Attributes:
        num_players (int): Total number of players in the game.
        starting_stack (int): Initial chip stack assigned to each player.
        first_dealer (Optional[int]): Index of the first dealer; randomly set if None.
        seed (Optional[int]): Seed for random number generation, ensuring reproducibility.
        current_idx (int): Index of the player whose turn it is to act.
        dealer_idx (int): Index of the current dealer.
        current_bet (int): Highest bet placed in the current betting round.
        pot (int): Total chips accumulated in the pot.
        betting_round (BettingRound): Current stage of the hand (e.g., PREFLOP, FLOP).
        hand_number (int): Number of hands played in the game so far.
        deck (SeededDeck): Deck of cards used for drawing, optionally seeded.
        players (List[Player]): List of player objects representing all participants.
        community_cards (List[Card]): List of community cards currently on the table.
    """

    # Game State
    num_players: int = 0
    starting_stack: int = 0
    first_dealer: Optional[int] = None
    seed: Optional[int] = None
    current_idx: int = 0
    dealer_idx: int = 0
    current_bet: int = 0
    pot: int = 0
    betting_round: BettingRound = BettingRound.START
    hand_number: int = 0
    deck: SeededDeck = field(default_factory=SeededDeck)
    players: List[PlayerState] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization method to set up the game state.
        If `first_dealer` is not specified, it is randomly selected from the range of player
        indices using the provided seed (if any). Sets `dealer_idx` to this value.
        """
        if self.first_dealer is None:
            self.first_dealer = np.random.default_rng(self.seed).integers(
                0, self.num_players - 1
            )
        self.dealer_idx = self.first_dealer

    def reset_for_new_hand(self):
        """Reset the game state to prepare for a new hand.
        Clears community cards, resets the pot and current bet to zero, shuffles the deck,
        and resets each player's hand and betting state for the new hand.
        """
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.deck.shuffle()
        for player in self.players:
            player.reset_for_new_hand()

    def reset(self, seed: Optional[int] = None):
        """Fully reset the game state to start a new game.
        Sets the betting round to START, resets the hand number to zero, reinitializes the
        dealer index to the first dealer, creates a new deck (seeded if `seed` is provided),
        instantiates new player objects with the starting stack, and calls `reset_for_new_hand`.
        Args:
            seed (Optional[int]): Seed for the deck’s random number generator, if provided.
        """
        self.betting_round = BettingRound.START
        self.hand_number = 0
        self.dealer_idx = self.first_dealer
        self.deck = SeededDeck(seed) if seed is not None else SeededDeck()
        self.players = [
            PlayerState(_idx=i, _chips=self.starting_stack) for i in range(self.num_players)
        ]
        self.reset_for_new_hand()

    def to_json(self) -> Dict[str, Any]:
        """Convert the game state to a JSON-compatible dictionary.
        Returns:
            A dictionary representation of the game state, including player states,
            community cards, and other game attributes.
        """
        return {
            "num_players": self.num_players,
            "starting_stack": self.starting_stack,
            "first_dealer": self.first_dealer,
            "seed": self.seed,
            "current_idx": self.current_idx,
            "dealer_idx": self.dealer_idx,
            "current_bet": self.current_bet,
            "pot": self.pot,
            "betting_round": self.betting_round.name,
            "hand_number": self.hand_number,
            "deck": [card_to_str(card) for card in self.deck.cards],
            "players": [player.to_json() for player in self.players],
            "community_cards": [card_to_str(card) for card in self.community_cards]
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PokerGameState':
        """Reconstruct the game state from a JSON dictionary.
        Args:
            data: A dictionary containing the serialized game state.
        Returns:
            An instance of PokerGameState with attributes populated from the dictionary.
        """
        players = [PlayerState.from_json(player_data) for player_data in data["players"]]
        deck = SeededDeck.from_cards([Card.new(card_str) for card_str in data["deck"]])
        community_cards = [Card.new(card_str) for card_str in data["community_cards"]]
        return cls(
            num_players=data["num_players"],
            starting_stack=data["starting_stack"],
            first_dealer=data["first_dealer"],
            seed=data.get("seed"),
            current_idx=data["current_idx"],
            dealer_idx=data["dealer_idx"],
            current_bet=data["current_bet"],
            pot=data["pot"],
            betting_round=BettingRound[data["betting_round"]],
            hand_number=data["hand_number"],
            deck=deck,
            players=players,
            community_cards=community_cards
        )