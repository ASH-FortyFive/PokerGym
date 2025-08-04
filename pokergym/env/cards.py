# Wrappers around deuces Deck to allow for seeding and reproducibility
from random import Random
from deuces import Deck
from deuces import Card


# Useful contants, this is the worst possible hand in Texas Hold'em in deuces
WORST_RANK = 7463

class SeededDeck(Deck):
    """A seeded deck that allows for reproducible shuffling.
    This class extends the deuces Deck class to allow for seeding
    and reproducibility of the deck's shuffle order.
    """
    def __init__(self, seed=None):
        self._random = Random(seed)
        self.shuffle()

    def shuffle(self):
        """Shuffle the deck using the seeded random number generator."""
        self.cards = Deck.GetFullDeck()
        self._random.shuffle(self.cards)

    def seed(self, seed):
        """Reset the random seed for reproducibility.
        Args:
            seed: The seed to use for random number generation.
        """
        self._random.seed(seed)

    def __len__(self):
        return len(self.cards)

    def __eq__(self, other):
        """Check equality with another SeededDeck."""
        if not isinstance(other, SeededDeck):
            return False
        return self.cards == other.cards    

    @classmethod
    def from_cards(cls, deck):
        """Create a SeededDeck from a list of deuces Card objects.
        Args:
            deck: A list of deuces Card objects to initialize the deck.
        """
        instance = cls()
        instance.cards = deck
        return instance

def card_to_int(card: Card) -> int:
    """Convert a deuces Card to low dim integer representation.
    0 is no card, 1-52 are the cards.
    1-13 are the ranks of Spades, 14-26 are Hearts,
    27-39 are Diamonds, and 40-52 are Clubs.
    Args:
        card: The deuces Card object to convert.
    Returns:
        An integer representation of the card, or 0 if card is None.
    """
    if card is None:
        return 0
    rank_int = Card.get_rank_int(card)
    suit_int = Card.get_suit_int(card) 
    suit_int = (suit_int & -suit_int).bit_length() - 1
    return rank_int + 13*suit_int + 1

def int_to_card(card_int: int) -> Card:
    """ Convert a low dim integer representation to a deuces Card.
    0 is no card, 1-52 are the cards.
    1-13 are the ranks of Spades, 14-26 are Hearts,
    27-39 are Diamonds, and 40-52 are Clubs.
    Args:
        card_int: The integer representation of the card.
    Returns:
        The corresponding deuces Card "object", or None if card_int is 0.
    """
    if card_int == 0:
        return None
    rank = (card_int - 1) % 13
    suit = (card_int - 1) // 13
    return Card.new(f"{Card.int_to_rank(rank)}{Card.int_to_suit(suit)}")    

def card_to_str(card: Card) -> str:
    """Convert a deuces Card to its string representation.
    Args:
        card: The deuces Card object to convert.
    Returns:
        A string representation of the card, or an empty string if card is None.
    """
    suit_int = Card.get_suit_int(card)
    rank_int = Card.get_rank_int(card)
    suit_str = Card.INT_SUIT_TO_CHAR_SUIT[suit_int]
    rank_str = Card.STR_RANKS[rank_int]
    string = f"{rank_str}{suit_str}"
    return string